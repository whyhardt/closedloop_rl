import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

import time
import copy
from typing import Optional, Union

# from rnn import BaseRNN, EnsembleRNN
from resources.rnn import BaseRNN, EnsembleRNN

class DatasetRNN(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor, batch_size: Optional[int] = None, device=torch.device('cpu')):
        """Initializes the dataset for training the RNN. Holds information about the previous actions and rewards as well as the next action.
        Actions can be either one-hot encoded or indexes.

        Args:
            xs (torch.Tensor): Actions and rewards in the shape (n_sessions, n_timesteps, n_features)
            ys (torch.Tensor): Next action
            batch_size (Optional[int], optional): Sets batch size if desired else uses n_samples as batch size.
            device (torch.Device, optional): Torch device. If None, uses cuda if available else cpu.
        """
        
        # check for type of xs and ys
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=torch.float32)
            
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        self.batch_size = batch_size if batch_size is not None else len(xs)
       
    def __len__(self):
        return self.xs.shape[0]
    
    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx, :]


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


def train_step(
    model: BaseRNN,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._Loss,
    n_steps: int = 10,
    ):
    
    # predict y
    model.initial_state(batch_size=len(x))
    for i in range(0, x.shape[1]-n_steps, n_steps):
        y_pred, _ = model(x[:, i:i+n_steps], model.get_state(detach=True), batch_first=True)
        
        y_target = y[:, i:i+n_steps]
        
        loss = 0
        for t in range(n_steps):
            loss += loss_fn(y_pred[:, t], y_target[:, t])
        loss /= n_steps
        
        if loss.requires_grad:
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model, optimizer, loss


def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    n_steps_per_call: int = None,
    ):

    """
    Trains a model for a fixed number of steps.
    """
    
    # start training
    model.initial_state(batch_size=len(xs))
    n_steps = n_steps_per_call if n_steps_per_call is not None else xs.shape[1]-1
    model, optimizer, loss = train_step(model, xs, ys, optimizer, loss_fn, n_steps)

    return model, optimizer, loss
    

def fit_model(
    model: Union[BaseRNN, EnsembleRNN],
    dataset: DatasetRNN,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-5,
    epochs: int = 1,
    n_steps_per_call: int = None,
    batch_size: int = None,
    sampling_replacement: bool = False,
    n_submodels: int = 1,
    return_ensemble: bool = False,
    voting_type: int = EnsembleRNN.MEAN,
):
    
    # initialize submodels
    if isinstance(model, BaseRNN):
        models = [model for _ in range(n_submodels)]
    elif isinstance(model, EnsembleRNN):
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

    # create loss list
    losses_over_time =[]   

    # start training
    while continue_training:
        try:
            t_start = time.time()
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
                        n_steps_per_call=n_steps_per_call,
                        # loss_fn = categorical_log_likelihood
                    )
                
                    loss += loss_i

                loss /= n_submodels

                
            
            if n_submodels > 1 and return_ensemble:
                model_backup = EnsembleRNN(models, voting_type=voting_type)
                optimizer_backup = optimizers
            else:
                # update model and optimizer via averaging over the parameters
                if n_submodels > 1:
                    avg_state_dict = average_parameters(models)
                    for submodel in models:
                        submodel.load_state_dict(avg_state_dict)
                model_backup = models[0]
                optimizer_backup = optimizers[0]
            
            # update training state
            n_calls_to_train_model += 1
            
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
            
            msg = f'Epoch {n_calls_to_train_model}/{epochs} --- Loss: {loss:.7f}; Time: {time.time()-t_start:.1f}s; Convergence value: {convergence_value:.2e}'
           
            if converged:
                msg += '\nModel converged!'
            elif n_calls_to_train_model >= epochs:
                msg += '\nMaximum number of training epochs reached.'
                if not converged:
                    msg += '\nModel did not converge yet.'
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'
        print(msg)

        # add loss to list
        losses_over_time.append(float(loss)) 

        
    if n_submodels > 1 and return_ensemble:
        model_backup = EnsembleRNN(models, voting_type=voting_type)
        optimizer_backup = optimizers
    else:
        model_backup = models[0]
        optimizer_backup = optimizers[0]
        
    return model_backup, optimizer_backup, losses_over_time
    