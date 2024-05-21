import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from typing import Any, Dict, Optional
import time

# sys.path.append('.')
from rnn import RNN


class DatasetRNN(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor, batch_size: Optional[int] = None, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # check for type of xs and ys
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=torch.float32)
            
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        self.batch_size = batch_size if batch_size is not None else len(xs)
       
    def __len__(self):
        return self.xs.shape[1]
    
    def __getitem__(self, idx):
        return self.xs[:, idx], self.ys[:, idx]


def categorical_log_likelihood(logits: torch.Tensor, target: torch.Tensor):
    # Mask any errors for which label is negative
    mask = torch.logical_not(target < 0)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_liks = target * log_probs
    masked_log_liks = torch.multiply(log_liks, mask)
    loss = -torch.nansum(masked_log_liks)/torch.prod(torch.tensor(masked_log_liks.shape[:-1]))
    return loss


def train_step(
    model: RNN,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    ):
    
    # predict y
    model.initial_state(batch_size=len(x), device=x.device)
    y_pred, _ = model(x, model.get_state(), batch_first=True)
    # y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    
    loss = 0
    for i in range(y.shape[1]):
        loss += loss_fn(y_pred[:, i], y[:, i])
    loss /= y.shape[1]
    # if y_pred.dim() == 3 and y.dim() == 2:
    #     y = y.unsqueeze(1)
    # loss = loss_fn(y_pred, y)
    
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model, optimizer, loss


def batch_train(
    model: RNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    # n_steps: int = 1000,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    ):

    """
    Trains a model for a fixed number of steps.
    """
    
    # initialize optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # start training
    # training_loss = []
    # loss = 0
    # t_start = time.time()
    model.initial_state(batch_size=len(xs), device=xs.device)
    # for step in range(1, n_steps):
    model, optimizer, loss = train_step(model, xs, ys, optimizer, loss_fn)
            
    return model, optimizer, loss
    

def fit_model(
    model: RNN,
    dataset: DatasetRNN,
    optimizer: torch.optim.Optimizer = None,
    rnn_state_dict: Optional[Dict[str, Any]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    convergence_threshold: float = 1e-5,
    # n_steps_per_call: int = -1,
    epochs: int = 10,
    batch_size: int = None,
):
    
    # check if n_steps_per_call is set otherwise set it to the length of the dataset
    # if n_steps_per_call == -1:
    #     n_steps_per_call = dataset.xs.shape[0]
    # else:
    #     if n_steps_per_call > dataset.xs.shape[0]:
    #         print(f'n_steps_per_call is larger than the number of steps in the dataset. Setting n_steps_per_call to {dataset.xs.shape[1]}.')
    #         n_steps_per_call = dataset.xs.shape[0]
    
    # initialize optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # load rnn state and optimizer state
    if rnn_state_dict is not None:
        model.load_state_dict(rnn_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    
    # initialize dataloader
    if batch_size is None:
        batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # initialize training
    continue_training = True
    converged = False
    loss = torch.inf
    n_calls_to_train_model = 0
    
    model_backup = model
    optimizer_backup = optimizer
        
    # start training
    while continue_training:
        try:
            # get next batch
            xs, ys = next(iter(dataloader))
            # train model
            t_start = time.time()
            model, optimizer, loss_new = batch_train(
                model=model,
                xs=xs,
                ys=ys,
                optimizer=optimizer,
                # loss_fn = categorical_log_likelihood
            )
            
            model_backup = model
            optimizer_backup = optimizer
            
            # update training state
            n_calls_to_train_model += 1
            
            # check for convergence
            convergence_value = torch.abs(loss - loss_new) / loss
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            
            if converged:
                msg = '\nModel converged!'
            elif n_calls_to_train_model >= epochs:
                msg = '\nMaximum number of training epochs reached.'
                if not converged:
                    msg += '\nModel did not converge yet.'
            else:
                msg = f'\nSteps {n_calls_to_train_model}/{epochs} --- Loss: {loss:.7f}; Time: {time.time()-t_start:.1f}s; Convergence value: {convergence_value:.2e} --- Continue training...'
                
            loss = loss_new
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'
        print(msg)
        
    return model_backup, optimizer_backup, loss