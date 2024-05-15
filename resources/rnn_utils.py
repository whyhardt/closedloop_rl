import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from typing import Any, Dict, Optional
import time

from rnn import RNN


class DatasetRNN(Dataset):
    def __init__(self, xs, ys, batch_size: Optional[int] = None):
        self.xs = xs
        self.ys = ys
        self.batch_size = batch_size if batch_size is not None else len(xs)
       
    def __len__(self):
        return len(self.xs)
    
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


def train_step(
    model: RNN,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    ):
    
    # predict y
    y_pred, _ = model(x)
    
    # compute loss
    loss = loss_fn(y_pred, y)
    
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
    n_steps: int = 1000,
    loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    ):

    """
    Trains a model for a fixed number of steps.
    """
    
    # initialize optimizer
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # start training
    training_loss = []
    loss = 0
    model.initital_state(batch_size=xs.shape[0])
    t_start = time.time()
    for step in range(n_steps):
        model, optimizer, loss = train_step(model, xs, ys, optimizer, loss_fn)
        
        # Log every 10th step
        if step % 10 == 9:
            training_loss.append(float(loss))
            print(f'\rStep {step + 1} of {n_steps}; '
            f'Loss: {loss:.7f}; '
            f'Time: {time.time()-t_start:.1f}s)', end='')
            
    return model, optimizer, loss/n_steps
    

def fit_model(
    model: RNN,
    dataset: DatasetRNN,
    optimizer: torch.optim.Optimizer = None,
    rnn_state_dict: Optional[Dict[str, Any]] = None,
    optimizer_state_dict: Optional[Dict[str, Any]] = None,
    convergence_threshold: float = 1e-5,
    n_steps_per_call: int = 500,
    n_steps_max: int = 2000,
    batch_size: int = None,
):
    
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # initialize training
    continue_training = True
    converged = False
    loss = torch.inf
    n_calls_to_train_model = 0
    
    # start training
    while continue_training:
        # get next batch
        xs, ys = next(iter(dataloader))
        # train model
        model, optimizer, loss_new = batch_train(
            model=model,
            xs=xs,
            ys=ys,
            optimizer=optimizer,
            n_steps=n_steps_per_call,
        )
        
        # update training state
        n_calls_to_train_model += 1
        
        # check for convergence
        convergence_value = torch.abs(loss - loss_new) / loss
        converged = convergence_value < convergence_threshold
        continue_training = not converged and n_calls_to_train_model*n_steps_per_call < n_steps_max
        
        if converged:
            msg = '\nModel converged!'
        elif n_calls_to_train_model*n_steps_per_call >= n_steps_max:
            msg = '\nMaximum number of training steps reached.'
            if not converged:
                msg += '\nModel did not converge yet.'
        else:
            msg = f'\nConvergence value: {convergence_value:.2e} --- Continue training...'
            
        loss = loss_new
        
        print(msg)
        
    return model, optimizer, loss