import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import copy
import numpy as np
from typing import Union, List, NamedTuple

# from rnn import BaseRNN, EnsembleRNN
from resources.rnn import BaseRNN, EnsembleRNN
from resources.rnn_utils import DatasetRNN

# ensemble types
class ensembleTypes(NamedTuple):
    NONE = -1
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
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    weight_reg_rnn: float = 0e0,
    reg_interval: int = 1,
    stride: int = 1,
    keep_predictions: bool = False,
    ):

    """
    Trains a model with the given batch.
    """
    
    if stride == -1:
        stride = xs.shape[1]
    
    if n_steps_per_call is None or n_steps_per_call == -1:
        n_steps_per_call = xs.shape[1]
    
    # predict y and compute loss
    model.initial_state(batch_size=len(xs))
    
    # state = model.get_state(detach=True)
    # for t in range(0, xs.shape[1], stride):
    #     n_steps = min(xs.shape[1]-t, n_steps_per_call)
    #     # get state for the next time step - adjust to stride parameter
    #     # _, state = model(xs[:, t], state, batch_first=True)
    #     # state_buffer = model.get_state(detach=True)
    #     y_pred = model(xs[:, t:t+n_steps], state, batch_first=True)[0]
        
    #     # compute prediction loss and optimize network
    #     if keep_predictions:
    #         loss = 0
    #         for i in range(y_pred.shape[1]):
    #             loss += loss_fn(y_pred[:, i], ys[:, t+i])
    #         loss /= i
    #     else:
    #         loss = loss_fn(y_pred[:, -1], ys[:, (t)+(n_steps-1)])
    #     if torch.is_grad_enabled():
            
    #         # null hypothesis penalty
    #         # reg_null = penalty_null_hypothesis(model, batch_size=128)
    #         # loss += weight_reg_rnn * reg_null
            
    #         # parameter optimization
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     # run model with updated parameters from start until t+stride to get the next state for the next iteration
    #     if t+stride < xs.shape[1]:
    #         with torch.no_grad():
    #             model.eval()
    #             model(xs[:, :t+1+stride], batch_first=True)
    #             model.train()
    #         state = model.get_state(detach=True)
    #     else:
    #         break
    #     # state = state_buffer

    # --------------------------------------------------------------
    # old procedure; may be too data inefficient due to big time steps
    # --------------------------------------------------------------
    
    # compute loss and optimize network w.r.t. rnn-predictions + null-hypothesis penalty
    for t in range(0, xs.shape[1], n_steps_per_call):
        n_steps = min(xs.shape[1]-t, n_steps_per_call)
        state = model.get_state(detach=True)
        y_pred = model(xs[:, t:t+n_steps], state, batch_first=True)[0][:, -1]
        loss = loss_fn(y_pred, ys[:, t+n_steps-1])
        
        if torch.is_grad_enabled():
            
            # --------------------------------------------------------------
            # 1st approach for training with regularizations
            # --------------------------------------------------------------
            
            # fit the regularization for each prediction iteration
            # null hypothesis penalty
            #reg_null = penalty_null_hypothesis(model, batch_size=128)
            #loss += weight_reg_rnn * reg_null
            
            # --------------------------------------------------------------
            # Leave this code block when switching between the two approaches
            # --------------------------------------------------------------
            
            # parameter optimization for prediction
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --------------------------------------------------------------
            # 2nd approach for training with regularizations
            # --------------------------------------------------------------
            
            # fit the regularization n times before fitting the prediction again
            for _ in range(reg_interval):
                reg_null = penalty_null_hypothesis(model, batch_size=128)
                loss_reg = weight_reg_rnn * reg_null
                # parameter optimization
                optimizer.zero_grad()
                loss_reg.backward()
                optimizer.step()
    
    return model, optimizer, loss.item()
    

def fit_model(
    model: Union[BaseRNN, EnsembleRNN],
    dataset: DatasetRNN,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-5,
    epochs: int = 1,
    batch_size: int = None,
    sampling_replacement: bool = False,
    n_submodels: int = 1,
    ensemble_type: int = ensembleTypes.NONE,
    voting_type: int = EnsembleRNN.MEDIAN,
    evolution_interval: int = None, verbose: bool = True,
    n_steps_per_call: int = None,
):
    
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
    
    # initialize dataloader
    if batch_size is None:
        batch_size = len(dataset)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if not sampling_replacement:
        # if no ensemble model is used, use normaler dataloader instance with sampling without replacement
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        # if ensemble model is used, use random sampler with replacement
        sampler = RandomSampler(dataset, replacement=True, num_samples=batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # initialize training    
    model_backup = model
    optimizer_backup = optimizers[0]
    
    def average_parameters(models):
        avg_state_dict = {key: None for key in models[0].state_dict().keys()}
        for key in avg_state_dict.keys():
            avg_state_dict[key] = torch.mean(torch.stack([model.state_dict()[key].data for model in models]), dim=0)
        return avg_state_dict
    
    loss = torch.inf
    if epochs == 0:
        continue_training = False
        msg = 'No training epochs specified. Model will not be trained.'
        if verbose:
            print(msg)
    else:
        continue_training = True
        converged = False
        n_calls_to_train_model = 0
        len_last_losses = min([20, epochs])
        last_losses = torch.ones((len_last_losses,))
        weights_losses = torch.linspace(0.5, 1, len_last_losses-1)
        sum_weights_losses = torch.sum(weights_losses)
    
    # start training
    while continue_training:
        try:
            t_start = time.time()
            n_calls_to_train_model += 1
            loss = 0
            if isinstance(model, EnsembleRNN):
                Warning('EnsembleRNN is not implemented for training yet. If you want to train an ensemble model, please train the submodels separately using the n_submodels argument and passing a single BaseRNN.')
                with torch.no_grad():
                    # get next batch
                    xs, ys = next(iter(dataloader))
                    xs = xs.to(models.device)
                    ys = ys.to(models.device)
                    # train model
                    _, _, loss = batch_train(
                        model=models,
                        xs=xs,
                        ys=ys,
                        optimizer=None,
                        n_steps_per_call=-1,
                        stride=-1,
                    )
            else:
                for i in range(len(models)):
                    # get next batch
                    xs, ys = next(iter(dataloader))
                    xs = xs.to(models[i].device)
                    ys = ys.to(models[i].device)
                    # train model
                    models[i], optimizers[i], loss_i = batch_train(
                        model=models[i],
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[i],
                        n_steps_per_call=n_steps_per_call,
                        keep_predictions=True,
                        # loss_fn = categorical_log_likelihood
                    )
                    loss += loss_i
                loss /= len(models)
            
            if n_submodels > 1 and ensemble_type == ensembleTypes.VOTE:
                model_backup = EnsembleRNN(models, voting_type=voting_type)
                optimizer_backup = optimizers
            else:
                if n_submodels > 1 and ensemble_type == ensembleTypes.AVERAGE:
                    avg_state_dict = average_parameters(models)
                    for submodel in models:
                        submodel.load_state_dict(avg_state_dict)
                model_backup = models[0]
                optimizer_backup = optimizers[0]
            
            if n_submodels > 1 and evolution_interval is not None and n_calls_to_train_model % evolution_interval == 0:
                # make sure that evolution interval is big enough so that the ensemble model can be trained effectively before evolution
                xs, ys = next(iter(dataloader))
                # check if current population is bigger than n_submodels (relevant for first evolution step)
                if len(models) > n_submodels:
                    n_best = n_submodels
                    n_children = 1
                else:
                    n_best, n_children = None, None
                models, optimizers = evolution_step(models, optimizers, xs.to(dataloader.dataset.device), ys.to(dataloader.dataset.device), n_best, n_children)
                n_submodels = len(models)
            
            if np.isinf(loss):
                print('smth wrong')
            
            # update last losses according to fifo principle
            last_losses[:-1] = last_losses[1:].clone()
            last_losses[-1] = copy.copy(loss)

            # check for convergence
            # convergence_value = torch.abs(loss - loss_new) / loss
            # compute convergence value based on weighted mean of last 100 losses with weight encoding the position in the list
            if n_calls_to_train_model < len_last_losses-1:
                convergence_value = (torch.sum(torch.abs(torch.diff(last_losses)[-n_calls_to_train_model:]) * weights_losses[-n_calls_to_train_model:]) / torch.sum(weights_losses[-n_calls_to_train_model:])).item()
                converged = False
                continue_training = True
            else:
                convergence_value = (torch.sum(torch.abs(torch.diff(last_losses)) * weights_losses) / sum_weights_losses).item()
                converged = convergence_value < convergence_threshold
                continue_training = not converged and n_calls_to_train_model < epochs
            
            msg = None
            if verbose:
                msg = f'Epoch {n_calls_to_train_model}/{epochs} --- Loss: {loss:.7f}; Time: {time.time()-t_start:.4f}s; Convergence value: {convergence_value:.2e}'                
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
        model_backup = models[0]
        optimizer_backup = optimizers[0]
        
    return model_backup, optimizer_backup, loss


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
            _, _, loss = batch_train(model, xs, ys, stride=-1, keep_predictions=True)
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