import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import time
import copy
import numpy as np
from typing import Union, List, NamedTuple

# from rnn import BaseRNN, EnsembleRNN
from resources.rnn import BaseRNN, EnsembleRNN
from resources.rnn_utils import DatasetRNN

# ensemble types
class ensembleTypes(NamedTuple):
    BEST = -1
    VOTE = 0
    AVERAGE = 1

class categorical_log_likelihood(nn.modules.loss._Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
    
    def forward(logits: torch.Tensor, target: torch.Tensor):
        # Mask any errors for which label is negative
        mask = torch.logical_not(target < 0)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_liks = target * log_probs
        masked_log_liks = torch.multiply(log_liks, mask)
        loss = -torch.nansum(masked_log_liks)/torch.prod(torch.tensor(masked_log_liks.shape[:-1]))
        return loss


def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    n_steps_per_call: int = -1,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),#nn.BCEWithLogitsLoss()#
    ):

    """
    Trains a model with the given batch.
    """
    
    if n_steps_per_call == -1:
        n_steps_per_call = xs.shape[1]
    
    # predict y and compute loss
    model.set_initial_state(batch_size=len(xs))
    state = model.get_state(detach=True)
    
    # compute loss and optimize network w.r.t. rnn-predictions + null-hypothesis penalty
    for t in range(0, xs.shape[1]-1, n_steps_per_call):
        n_steps = min(xs.shape[1]-t-1, n_steps_per_call)
        state = model.get_state(detach=True)
        y_pred, _, vae_outputs = model(xs[:, t:t+n_steps], state, batch_first=True)
        
        reconstruction_loss = torch.tensor(0., device=y_pred.device)
        kl_div = torch.tensor(0., device=y_pred.device)
        for i in range(n_steps):
            reconstruction_loss += loss_fn(y_pred[:, i], ys[:, t+i])
            kl_div += torch.mean(0.5 * torch.sum(torch.exp(vae_outputs[:, i, :, 1]) + torch.square(vae_outputs[:, i, :, 0]) - 1 - vae_outputs[:, i, :, 1], dim=-1))
        reconstruction_loss /= n_steps
        kl_div /= n_steps          
        
        loss = reconstruction_loss + kl_div #* 1e1
        
        if torch.is_grad_enabled():
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()  # retain_graph=True
            optimizer.step()
    
    return model, optimizer, (loss.item(), reconstruction_loss.item(), kl_div.item())
    

def fit_model(
    model: Union[BaseRNN, EnsembleRNN],
    dataset_train: DatasetRNN = None,
    dataset_test: DatasetRNN = None,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-5,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    n_oversampling: int = -1,
    n_submodels: int = 1,
    ensemble_type: int = ensembleTypes.BEST,
    voting_type: int = EnsembleRNN.MEDIAN,
    evolution_interval: int = None,
    n_steps_per_call: int = -1,
    verbose: bool = True,
):
    
    # make sure that either epochs and training dataset or at least a test dataset are given
    if dataset_train == None and epochs > 0:
        raise Warning('dataset_train is None but epochs are higher than 0. Setting epochs to 0')
        epochs = 0
    if dataset_train == None and dataset_test == None:
        raise ValueError('dataset_train and dataset_test are both None. Give at least one dataset.')
    
    # initialize submodels
    if isinstance(model, BaseRNN):
        if n_submodels == 1:
            models = [model]
        else:
            models = [copy.deepcopy(model) for _ in range(n_submodels)]
    elif isinstance(model, EnsembleRNN):
        models = model
    elif isinstance(model, list):
        models = model
    else:
        raise ValueError('Model must be either a BaseRNN (can be trained) or an EnsembleRNN (can only be tested).')
    
    # initialize optimizer
    optimizers = [torch.optim.Adam(submodel.parameters(), lr=1e-3) for submodel in models]
    if optimizer is None:
        pass
    elif isinstance(optimizer, torch.optim.Optimizer):
        for subopt in optimizers:
            subopt.load_state_dict(optimizer.state_dict())
    elif isinstance(optimizer, list) and len(optimizer) == len(models):
        optimizers = optimizer
    else:
        raise ValueError('Optimizer must be either of NoneType, a single optimizer or a list of optimizers with the same length as the number of submodels.')
    
    # set up learning rate scheduler for each optimizer
    # scheduler = [ReduceLROnPlateau(optim, patience=1e2) for optim in optimizers]
    
    if dataset_train is not None:
        # initialize dataloader
        if batch_size == -1:
            batch_size = len(dataset_train)//n_submodels
        # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if bagging:
            # use random sampling with replacement
            if n_oversampling == -1:
                n_oversampling = batch_size
            sampler = RandomSampler(dataset_train, replacement=True, num_samples=n_oversampling)
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler)
        else:
            # use normal dataloader instance with sampling without replacement
            dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        batch_iteration_constant = len(models) if len(models) > 1 else len(dataset_train) // batch_size
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    
    # initialize training    
    model_backup = model
    optimizer_backup = optimizers[0]
    
    def average_parameters(models):
        
        averaged_state_dict = copy.deepcopy(models[0].state_dict())

        # Sum the parameters of all submodels
        for key in averaged_state_dict:
            averaged_state_dict[key].detach().zero_()
            for model in models:
                averaged_state_dict[key] += model.state_dict()[key]
            if model.state_dict()[key].dtype in (torch.int64, torch.int32):
                averaged_state_dict[key] = averaged_state_dict[key] // len(models)
            else:
                averaged_state_dict[key] /= len(models)
        
        return averaged_state_dict
    
    if epochs == 0:
        continue_training = False
        msg = 'No training epochs specified. Model will not be trained.'
        if verbose:
            print(msg)
    else:
        continue_training = True
        converged = False
        n_calls_to_train_model = 0
        convergence_value = 1
        last_loss = 1
        recency_factor = 0.5
    
    loss_train = 0
    loss_test = 0
    
    # start training
    while continue_training:
        try:
            loss_train = 0
            loss_train_rec = 0
            loss_train_kl = 0
            loss_test = 0
            loss_test_rec = 0
            loss_test_kl = 0
            t_start = time.time()
            n_calls_to_train_model += 1
            if isinstance(model, EnsembleRNN):
                Warning('EnsembleRNN is not implemented for training yet. If you want to train an ensemble model, please train the submodels separately using the n_submodels argument and passing a single BaseRNN.')
                with torch.no_grad():
                    # get next batch
                    xs, ys = next(iter(dataloader_train))
                    xs = xs.to(models.device)
                    ys = ys.to(models.device)
                    # train model
                    _, _, loss_train = batch_train(
                        model=models,
                        xs=xs,
                        ys=ys,
                        optimizer=None,
                        stride=-1,
                    )
            else:
                for i in range(batch_iteration_constant):
                    i_model = i if len(models) > 1 else 0
                    # get next batch
                    xs, ys = next(iter(dataloader_train))
                    xs = xs.to(models[i_model].device)
                    ys = ys.to(models[i_model].device)
                    # train model
                    models[i_model], optimizers[i_model], loss_i = batch_train(
                        model=models[i_model],
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[i_model],
                        n_steps_per_call=n_steps_per_call,
                        # keep_predictions=True,
                        # loss_fn = categorical_log_likelihood
                    )
                    loss_train += loss_i[0]
                    loss_train_rec += loss_i[1]
                    loss_train_kl += loss_i[2]
                loss_train /= batch_iteration_constant
                loss_train_rec /= batch_iteration_constant
                loss_train_kl /= batch_iteration_constant
            
            if len(models) > 1 and ensemble_type == ensembleTypes.VOTE:
                model_backup = EnsembleRNN(models, voting_type=voting_type)
                optimizer_backup = optimizers
            else:
                if len(models) > 1 and ensemble_type == ensembleTypes.AVERAGE:
                    avg_state_dict = average_parameters(models)
                    for submodel in models:
                        submodel.load_state_dict(avg_state_dict)
                model_backup = models[0]
                optimizer_backup = optimizers[0]
            
            if len(models) > 1 and evolution_interval is not None and n_calls_to_train_model % evolution_interval == 0:
                # make sure that evolution interval is big enough so that the ensemble model can be trained effectively before evolution
                xs, ys = next(iter(dataloader_train))
                # check if current population is bigger than n_submodels (relevant for first evolution step)
                if len(models) > n_submodels:
                    n_best = n_submodels
                    n_children = 1
                else:
                    n_best, n_children = None, None
                models, optimizers = evolution_step(models, optimizers, xs.to(dataloader_train.dataset.device), ys.to(dataloader_train.dataset.device), n_best, n_children)
                n_submodels = len(models)
            
            if not isinstance(model, EnsembleRNN) and dataset_test is not None:
                models[0].eval()
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    xs = xs.to(models[0].device)
                    ys = ys.to(models[0].device)
                    # train model
                    _, _, loss_test = batch_train(
                        model=models[0],
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[0],
                        # loss_fn = categorical_log_likelihood
                    )
                models[0].train()
                loss_test_rec = loss_test[1]
                loss_test_kl = loss_test[2]
                loss_test = loss_test[0]

            # adapt learning rate based on validation loss
            # last_lr = 0
            # for s in scheduler:
            #     s.step(loss_test)
            #     last_lr += s.get_last_lr()[-1]
            # last_lr /= len(scheduler)
            
            # check for convergence
            dloss = last_loss - loss_test if dataset_test is not None else last_loss - loss_train
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold #or last_lr < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = 0
            last_loss += loss_test if dataset_test is not None else loss_train
            
            msg = None
            if verbose:
                msg = f'Epoch {n_calls_to_train_model}/{epochs} --- L(Training): {loss_train:.7f}'                
                if dataset_test is not None:
                    msg += f'; L(Validation): {loss_test:.7f}'
                msg += f'; Time: {time.time()-t_start:.4f}s; Convergence: {convergence_value:.2e}'
                # msg += f'; Learning rate: {last_lr:.2e}'
                msg += f'; L(Reconstr): {loss_test_rec if dataset_test is not None else loss_train_rec:.4f}'
                msg += f'; L(KL): {loss_test_kl if dataset_test is not None else loss_train_kl:.4f}'
                if converged:
                    msg += '\nModel converged!'
                elif n_calls_to_train_model >= epochs:
                    msg += '\nMaximum number of training epochs reached.'
                    if not converged:
                        msg += '\nModel did not converge yet.'
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'
        if verbose:
            print(msg)
        
    if n_submodels > 1 and ensemble_type == ensembleTypes.VOTE:
        model_backup = EnsembleRNN(models, voting_type=voting_type, device=models[0].device)
        optimizer_backup = optimizers
    else:
        if n_submodels > 1 and ensemble_type == ensembleTypes.AVERAGE:
            avg_state_dict = average_parameters(models)
            models[0].load_state_dict(avg_state_dict)
        model_backup = models[0]
        optimizer_backup = optimizers[0]
        
    return model_backup, optimizer_backup, loss_test if dataset_test is not None else loss_train


def evolution_step(models: List[nn.Module], optimizers: List[torch.optim.Optimizer], xs, ys, n_best: int = None, n_children: int = None):
    """Make an evolution step for the ensemble model by selecting the best models and creating children from them.

    Args:
        models (List[nn.Module]): _description_
        data (DatasetRNN): _description_
        n_best (int, optional): _description_. Defaults to 10.
        n_children (int, optional): _description_. Defaults to 10.
    """
    
    if n_best is None:
        n_best = len(models)//2
    elif len(models) < n_best:
        n_best = len(models)//2
    if n_children is None:
        n_children = 2
    elif len(models) < n_children*n_best:
        n_children = len(models)//n_best
    
    # select best models
    losses = torch.zeros(len(models))
    for i, model in enumerate(models):
        with torch.no_grad():
            _, _, loss = batch_train(model, xs, ys)
            losses[i] = loss

    # sort models by loss
    sorted_indices = torch.argsort(losses)
    best_models = [models[i] for i in sorted_indices[:n_best]]
    optimizers = [optimizers[i] for i in sorted_indices[:n_best]]
    
    # create children - no mutation because children will be trained on different data which should be enough to introduce diversity
    children = []
    children_optims = []
    for i in range(n_children):
        for model, optim in zip(best_models, optimizers):
            children.append(copy.deepcopy(model))
            children_optims.append(copy.deepcopy(optim))

    return children, children_optims


def penalty_null_hypothesis(model, batch_size: int = 1):
    """Compute a penalty for each subnetwork of the model based on the output of the corresponding subnetwork.
    This penalty serves as a regularization term to enforce the prior that this layer is not needed in the first place --> aka null-hypothesis.
    In this manner implemented hypothesis could be tested without model comparison but by the null-hypothesis."""
    
    reg_rnn = torch.zeros((), device=model.device)
    i = 0
    # create a random variable which is applicable for all subnetworks
    epsilon = torch.randn((batch_size, 32), device=model.device)
    for name, module in model.named_modules():
        if name.startswith('x') and not '.' in name:
            reg_rnn += torch.sum(torch.pow(module(epsilon[:, :module[0].in_features]), 2))/batch_size
            i += 1
    return reg_rnn/i


def penalty_correlated_update(model, batch_size: int = 1):
    """Compute a penalty for each subnetwork based on the influence of one input onto another output e.g. dxi/dxj = 0 for i != j.

    Args:
        model (BaseRNN): Model to test
    """
    
    reg_rnn = torch.zeros((1, 1), device=model.device)
    i = 0
    # create a random variable which is applicable for all subnetworks
    epsilon = torch.randn((batch_size, 32), device=model.device)
    for name, module in model.named_modules():
        if name.startswith('x') and not '.' in name:
            for j in range(module[-1].out_features):
                epsilon_j = epsilon.clone()
                epsilon_j[:, j] = 0
                for jj in range(module[-1].out_features):
                    if jj != j:
                        reg_rnn += torch.sum(torch.pow(module(epsilon[:, :module[0].in_features])[:, jj] - module(epsilon_j[:, :module[0].in_features])[:, jj], 2))/batch_size
                        i += 1
    
    return reg_rnn/i

def penalty_beta_range(model):
    """Compute the penalty for the network resulting in Q-Values higher than 1 and lower than 0."""
    
    return model.beta