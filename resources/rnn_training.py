import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from typing import Any, Dict, Optional
import time

from rnn import HybRNN


class DatasetRNN(Dataset):
    def __init__(self, xs: torch.Tensor, ys: torch.Tensor, batch_size: Optional[int] = None, device=None):
        """Initializes the dataset for training the RNN. Holds information about the previous actions and rewards as well as the next action.
        Actions can be either one-hot encoded or indexes.

        Args:
            xs (torch.Tensor): Actions and rewards in the shape (n_sessions, n_timesteps, n_features)
            ys (torch.Tensor): Next action
            batch_size (Optional[int], optional): Sets batch size if desired else uses n_samples as batch size.
            device (torch.Device, optional): Torch device. If None, uses cuda if available else cpu.
        """
        
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
    model: HybRNN,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.modules.loss._Loss,
    ):
    
    # predict y
    model.initial_state(batch_size=len(x), device=x.device)
    y_pred, _ = model(x, model.get_state(), batch_first=True)
    # y_pred = torch.nn.functional.softmax(y_pred, dim=1)
    
    loss = 0
    if loss_fn.__class__ == nn.CrossEntropyLoss().__class__:
        for i in range(y.shape[1]):
            loss += loss_fn(y_pred[:, i], y[:, i])
        loss /= y.shape[1]
    elif loss_fn.__class__ == categorical_log_likelihood.__class__:
        loss = loss_fn(y_pred, y)
    else:
        raise NotImplementedError('Loss function not implemented.')
    
    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return model, optimizer, loss


def batch_train(
    model: HybRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
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
    model: HybRNN,
    dataset: DatasetRNN,
    optimizer: torch.optim.Optimizer = None,
    rnn_state_dict: Optional[Dict[str, Any]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    convergence_threshold: float = 1e-5,
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
    
    len_last_losses = 20
    last_losses = torch.ones((len_last_losses,))
    weights_losses = torch.linspace(0, 1, len_last_losses-1)
    sum_weights_losses = torch.sum(weights_losses)
    
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
            
            # update last losses according fifo principle
            last_losses[:-1] = last_losses[1:].clone()
            last_losses[-1] = loss_new.item()
            
            # check for convergence
            # convergence_value = torch.abs(loss - loss_new) / loss
            # compute convergence value based on weighted mean of last 100 losses with weight encoding the position in the list
            if n_calls_to_train_model < len_last_losses-1:
                convergence_value = (torch.sum(torch.abs(torch.diff(last_losses)[-n_calls_to_train_model:]) * weights_losses[-n_calls_to_train_model:]) / torch.sum(weights_losses[-n_calls_to_train_model:])).item()
                converged = False
                continue_training = True
            else:
                convergence_value = (torch.sum(torch.abs(torch.diff(last_losses)) * weights_losses) / sum_weights_losses).item()
                converged = convergence_value < convergence_threshold #and loss_new.item() < convergence_threshold*2
                continue_training = not converged and n_calls_to_train_model < epochs
            
            if converged:
                msg = '\nModel converged!'
            elif n_calls_to_train_model >= epochs:
                msg = '\nMaximum number of training epochs reached.'
                if not converged:
                    msg += '\nModel did not converge yet.'
            else:
                msg = f'Epoch {n_calls_to_train_model}/{epochs} --- Loss: {loss:.7f}; Time: {time.time()-t_start:.1f}s; Convergence value: {convergence_value:.2e} --- Continue training...'
                
            loss = loss_new
        except KeyboardInterrupt:
            continue_training = False
            msg = 'Training interrupted. Continuing with further operations...'
        print(msg)
        
    return model_backup, optimizer_backup, loss