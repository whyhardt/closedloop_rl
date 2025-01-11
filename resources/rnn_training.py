import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from resources.rnn import BaseRNN
from resources.rnn_utils import DatasetRNN


def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    n_steps_per_call: int = -1,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    ):

    """
    Trains a model with the given batch.
    """
    
    if n_steps_per_call == -1:
        n_steps_per_call = xs.shape[1]
    
    model.set_initial_state(batch_size=len(xs))
    state = model.get_state(detach=True)
    
    loss_batch = 0
    iterations = 0
    for t in range(0, xs.shape[1], n_steps_per_call):
        n_steps = min(xs.shape[1]-t, n_steps_per_call)
        xs_step = xs[:, t:t+n_steps]
        ys_step = ys[:, t:t+n_steps]
        
        state = model.get_state(detach=True)
        y_pred = model(xs_step, state, batch_first=True)[0]
        
        mask = xs_step[:, :, :model._n_actions] > -1
        loss = loss_fn(
            (y_pred*mask).reshape(-1, model._n_actions), 
            (ys_step*mask).reshape(-1, model._n_actions)
            )
        
        loss_batch += loss
        iterations += 1
        
        # TODO: implement warm-up?
        
        if torch.is_grad_enabled():
            
            # L1 weight decay to encourage sparsification in the network
            loss += 1e-4*sum(torch.sum(torch.abs(param)) for param in model.parameters())
            
            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            # loss_batch.backward()
            optimizer.step()
    
    return model, optimizer, loss_batch.item()/iterations
    

def fit_model(
    model: BaseRNN,
    dataset_train: DatasetRNN,
    dataset_test: DatasetRNN = None,
    optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-5,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    n_steps_per_call: int = -1,
    verbose: bool = True,
    ):
    
    # initialize dataloader
    if batch_size == -1:
        batch_size = len(dataset_train)
    # use random sampling with replacement
    sampler = RandomSampler(dataset_train, replacement=True) if bagging else None
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False)
    if dataset_test is not None:
        dataloader_test = DataLoader(dataset_test, batch_size=len(dataset_test))
    
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
    iterations_per_epoch = len(dataset_train) // batch_size
    
    # start training
    while continue_training:
        try:
            loss_train = 0
            loss_test = 0
            t_start = time.time()
            n_calls_to_train_model += 1
            for _ in range(iterations_per_epoch):
                # get next batch
                xs, ys = next(iter(dataloader_train))
                xs = xs.to(model.device)
                ys = ys.to(model.device)
                # train model
                model, optimizer, loss_i = batch_train(
                    model=model,
                    xs=xs,
                    ys=ys,
                    optimizer=optimizer,
                    n_steps_per_call=n_steps_per_call,
                )
                loss_train += loss_i
            loss_train /= iterations_per_epoch
            
            if dataset_test is not None:
                model.eval()
                with torch.no_grad():
                    xs, ys = next(iter(dataloader_test))
                    xs = xs.to(model.device)
                    ys = ys.to(model.device)
                    # evaluate model
                    _, _, loss_test = batch_train(
                        model=model,
                        xs=xs,
                        ys=ys,
                        optimizer=optimizer,
                    )
                model.train()

            # check for convergence
            dloss = last_loss - loss_test if dataset_test is not None else last_loss - loss_train
            convergence_value += recency_factor * (np.abs(dloss) - convergence_value)
            converged = convergence_value < convergence_threshold
            continue_training = not converged and n_calls_to_train_model < epochs
            last_loss = 0
            last_loss += loss_test if dataset_test is not None else loss_train
            
            msg = None
            if verbose:
                msg = f'Epoch {n_calls_to_train_model}/{epochs} --- L(Training): {loss_train:.7f}'                
                if dataset_test is not None:
                    msg += f'; L(Validation): {loss_test:.7f}'
                msg += f'; Time: {time.time()-t_start:.2f}s; Convergence value: {convergence_value:.2e}'
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
            
    return model, optimizer, loss_train