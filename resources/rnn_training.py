import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import copy
import numpy as np
from typing import Union, List

import pysindy as ps

# from rnn import BaseRNN, EnsembleRNN
from resources.rnn import BaseRNN, EnsembleRNN
from resources.rnn_utils import DatasetRNN
from resources.sindy_utils import sindy_loss_x, create_dataset, constructor_update_rule_sindy
from resources.sindy_training import fit_model as fit_sindy_model, library_setup, datafilter_setup, setup_sindy_agent
from resources.bandits import AgentNetwork, AgentSindy, EnvironmentBanditsDrift

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
    n_steps_per_call: int = 16,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    sindy_ae: bool = False,
    weight_rnn_x: float = 1,
    weight_sindy_x: float = 1e-2,
    weight_sindy_reg: float = 1e-2,
    ):

    """
    Trains a model with the given batch.
    """
    
    # predict y
    model.initial_state(batch_size=len(xs))
    for t in range(0, xs.shape[1], n_steps_per_call):
        n_steps = min(xs.shape[1]-t, n_steps_per_call)
        state = model.get_state(detach=True)
        y_pred = model(xs[:, t:t+n_steps], state, batch_first=True)[0][:, -1]
        loss = loss_fn(y_pred, ys[:, t+n_steps-1])  # rnn loss in x-coordinates
        loss_sindy_x, loss_sindy_weights = None, None
        if sindy_ae:
            # train sindy
            z_train, control, feature_names = create_dataset(AgentNetwork(model, model._n_actions, model.device), xs, 200, 1, normalize=True)
            sindy = fit_sindy_model(z_train, control, feature_names, library_setup=library_setup, filter_setup=datafilter_setup)
            update_rule_sindy = constructor_update_rule_sindy(sindy)
            agent_sindy = setup_sindy_agent(update_rule_sindy, model._n_actions)
            
            # sindy loss in x-coordinates
            loss_sindy_x = sindy_loss_x(agent_sindy, DatasetRNN(xs[0], ys[0]))
            # sindy loss in z-coordinates --> Try first without; I dont think it is necessary since we are working with really low-dimensional input data for the RNN
            # sindy weight regularization
            loss_sindy_weights = torch.mean((torch.tensor([sindy[key].coefficients() for key in sindy])))
        
            loss = weight_rnn_x * loss + weight_sindy_x * loss_sindy_x + weight_sindy_reg * loss_sindy_weights
            
        if loss.requires_grad:
            # backpropagation
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
    
    return model, optimizer, loss, (loss_sindy_x, loss_sindy_weights)
    

def fit_model(
    model: Union[BaseRNN, EnsembleRNN],
    dataset: DatasetRNN,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-5,
    epochs: int = 1,
    batch_size: int = None,
    sampling_replacement: bool = False,
    n_submodels: int = 1,
    return_ensemble: bool = False,
    voting_type: int = EnsembleRNN.MEDIAN,
    sindy_ae: bool = False,
    evolution_interval: int = None, verbose: bool = True,
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
    if optimizer is not None:
        for subopt in optimizers:
            subopt.load_state_dict(optimizer.state_dict())
    
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
    if evolution_interval is not None:
        dataloader_evolution = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialize training
    continue_training = True
    converged = False
    loss = torch.inf
    n_calls_to_train_model = 0
    
    model_backup = model
    optimizer_backup = optimizers[0]
    
    len_last_losses = min([20, epochs])
    last_losses = torch.ones((len_last_losses,))
    weights_losses = torch.linspace(0.5, 1, len_last_losses-1)
    sum_weights_losses = torch.sum(weights_losses)
    
    def average_parameters(models):
        avg_state_dict = {key: None for key in models[0].state_dict().keys()}
        for key in avg_state_dict.keys():
            avg_state_dict[key] = torch.mean(torch.stack([model.state_dict()[key].data for model in models]), dim=0)
        return avg_state_dict
    
    # start training
    while continue_training:
        try:
            t_start = time.time()
            n_calls_to_train_model += 1
            loss = 0
            loss_sindy_x = 0
            loss_sindy_weights = 0
            if isinstance(model, EnsembleRNN):
                Warning('EnsembleRNN is not implemented for training yet. If you want to train an ensemble model, please train the submodels separately using the n_submodels argument and passing a single BaseRNN.')
                with torch.no_grad():
                    # get next batch
                    xs, ys = next(iter(dataloader))
                    # train model
                    _, _, loss, _ = batch_train(
                        model=models,
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[0],
                    )
            else:
                for i in range(n_submodels):
                    # get next batch
                    xs, ys = next(iter(dataloader))
                    # train model
                    models[i], optimizers[i], loss_i, loss_sindy_i = batch_train(
                        model=models[i],
                        xs=xs,
                        ys=ys,
                        optimizer=optimizers[i],
                        sindy_ae=sindy_ae,
                        # loss_fn = categorical_log_likelihood
                    )
                
                    loss += loss_i
                    if sindy_ae:
                        loss_sindy_x += loss_sindy_i[0]
                        loss_sindy_weights += loss_sindy_i[1]
                loss /= n_submodels
                if sindy_ae:
                    loss_sindy_x /= n_submodels
                    loss_sindy_weights /= n_submodels
            
            if n_submodels > 1 and return_ensemble:
                model_backup = EnsembleRNN(models, voting_type=voting_type)
                optimizer_backup = optimizers
            else:
                # update model and optimizer via averaging over the parameters
                if n_submodels > 1 and evolution_interval is None:
                    avg_state_dict = average_parameters(models)
                    for submodel in models:
                        submodel.load_state_dict(avg_state_dict)
                model_backup = models[0]
                optimizer_backup = optimizers[0]
            
            if n_submodels > 1 and evolution_interval is not None and n_calls_to_train_model % evolution_interval == 0:
                # make sure that evolution interval is big enough so that the ensemble model can be trained effectively before evolution
                models, optimizers = evolution_step(models, optimizers, DatasetRNN(*next(iter(dataloader_evolution))), 8, 2)
            
            # update last losses according fifo principle
            last_losses[:-1] = last_losses[1:].clone()
            last_losses[-1] = loss.item()

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
                if sindy_ae:
                    msg += f' --- Loss SINDy x: {loss_sindy_x:.7f}; Loss SINDy weights: {loss_sindy_weights:.7f}'
                
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
        
    if n_submodels > 1 and return_ensemble:
        model_backup = EnsembleRNN(models, voting_type=voting_type)
        optimizer_backup = optimizers
    else:
        model_backup = models[0]
        optimizer_backup = optimizers[0]
        
    return model_backup, optimizer_backup, loss


def evolution_step(models: List[nn.Module], optimizers: List[torch.optim.Optimizer], data: DatasetRNN, n_best=10, n_children=10):
    """Make an evolution step for the ensemble model by selecting the best models and creating children from them.

    Args:
        models (List[nn.Module]): _description_
        data (DatasetRNN): _description_
        n_best (int, optional): _description_. Defaults to 10.
        n_children (int, optional): _description_. Defaults to 10.
    """
    
    if len(models) < n_best:
        n_best = len(models)//2
    if len(models) < n_children*n_best:
        n_children = len(models)//n_best
        
    # select best models
    losses = torch.zeros(len(models))
    for model in models:
        with torch.no_grad():
            for i, model in enumerate(models):
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
    