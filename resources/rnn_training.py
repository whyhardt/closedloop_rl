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
class ensemble_types(NamedTuple):
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
    n_steps_per_call: int = None,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    weight_reg_rnn: float = 1e-2,
    ):

    """
    Trains a model with the given batch.
    """
    
    if n_steps_per_call is None:
        n_steps_per_call = xs.shape[1]
    
    # predict y and compute loss
    model.initial_state(batch_size=len(xs))
        
    # compute loss and optimize network w.r.t. rnn-predictions + null-hypothesis penalty
    for t in range(0, xs.shape[1], n_steps_per_call):
        n_steps = min(xs.shape[1]-t, n_steps_per_call)
        state = model.get_state(detach=True)
        y_pred = model(xs[:, t:t+n_steps], state, batch_first=True)[0][:, -1]
        loss = loss_fn(y_pred, ys[:, t+n_steps-1])  # rnn loss in x-coordinates
        
        if torch.is_grad_enabled():
            
            # reg_null = penalty_null_hypothesis(model, batch_size=128)   # null hypothesis penalty
            # loss += weight_reg_rnn * reg_null
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
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
    ensemble_type: int = -1,
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
    elif isinstance(optimizer, list) and len(optimizer) == n_submodels:
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
                    # train model
                    _, _, loss = batch_train(
                        model=models,
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[0],
                        n_steps_per_call=n_steps_per_call,
                    )
            else:
                for i in range(n_submodels):
                    # get next batch
                    xs, ys = next(iter(dataloader))
                    # train model
                    models[i], optimizers[i], loss_i = batch_train(
                        model=models[i],
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[i],
                        n_steps_per_call=n_steps_per_call
                        # loss_fn = categorical_log_likelihood
                    )
                
                    loss += loss_i
                loss /= n_submodels
            
            if n_submodels > 1 and ensemble_type == ensemble_types.VOTE:
                model_backup = EnsembleRNN(models, voting_type=voting_type)
                optimizer_backup = optimizers
            else:
                if n_submodels > 1 and ensemble_type == ensemble_types.AVERAGE:
                    avg_state_dict = average_parameters(models)
                    for submodel in models:
                        submodel.load_state_dict(avg_state_dict)
                model_backup = models[0]
                optimizer_backup = optimizers[0]
            
            if n_submodels > 1 and evolution_interval is not None and n_calls_to_train_model % evolution_interval == 0:
                # make sure that evolution interval is big enough so that the ensemble model can be trained effectively before evolution
                models, optimizers = evolution_step(models, optimizers, DatasetRNN(*next(iter(dataloader)), device=model[0].device))
                n_submodels = len(models)
            
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
        
    if n_submodels > 1 and ensemble_type == ensemble_types.VOTE:
        model_backup = EnsembleRNN(models, voting_type=voting_type)
        optimizer_backup = optimizers
    else:
        model_backup = models[0]
        optimizer_backup = optimizers[0]
        
    return model_backup, optimizer_backup, loss


def evolution_step(models: List[nn.Module], optimizers: List[torch.optim.Optimizer], data: DatasetRNN, n_best: int = None, n_children: int = None):
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
            _, _, loss = fit_model(model, data, verbose=False)
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
    """Compute a penalty for each subnetwork of the model based on the difference between the input and output of the layer.
    This penalty serves as a regularization term to enforce the prior that this layer is not needed in the first place.
    In this manner implemented hypothesis could be tested without model comparison but by the null-hypothesis."""
    
    reg_rnn = torch.zeros((), device=model.device)
    i = 0
    # create a random variable which is applicable for all subnetworks
    epsilon = torch.randn((batch_size, 32), device=model.device)
    for name, module in model.named_modules():
        if name.startswith('x') and not '.' in name:
            # if module[-1].out_features == module[0].in_features:
            #     reg_rnn += torch.sum(torch.pow(module(epsilon) - epsilon, 2))
            # elif module[-1].out_features < module[0].in_features:
            #     n_control_inputs = module[0].in_features - module[-1].out_features
            #     reg_rnn += torch.sum(torch.pow(module(epsilon) - epsilon[0, :-n_control_inputs], 2))
            # else:
            #     raise ValueError('Output features of the RNN-subnetwork must be less or equal to input features.')
            
            # All updates in the RNN are conceptualized in such a way that they are added onto the old value.
            # That means that the output of the RNN should be zero such that the null hypothesis is fulfilled.
            # I do not want no change to the input but the output to be zero!
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