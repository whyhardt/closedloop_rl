import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

import time
import copy
from typing import Optional, Union, Callable

import pysindy as ps

# from rnn import BaseRNN, EnsembleRNN
from resources.rnn import BaseRNN, EnsembleRNN

class DatasetRNN(Dataset):
    def __init__(
        self, 
        xs: torch.Tensor, 
        ys: torch.Tensor, 
        sequence_length: int = None,
        stride: int = 1,
        device=torch.device('cpu'),
        ):
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
        
        self.xs = xs
        self.ys = ys
        self.sequence_length = sequence_length if sequence_length is not None else xs.shape[1]
        self.stride = stride
        
        self.set_sequences()
        
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        
    def set_sequences(self):
        # sets sequences of length sequence_length with specified stride from the dataset
        xs_sequences = []
        ys_sequences = []
        for i in range(0, max(1, self.xs.shape[1]-self.sequence_length), self.stride):
            xs_sequences.append(self.xs[:, i:i+self.sequence_length, :])
            ys_sequences.append(self.ys[:, i:i+self.sequence_length, :])
        self.xs = torch.cat(xs_sequences, dim=0)
        self.ys = torch.cat(ys_sequences, dim=0)
        
        if len(self.xs.shape) == 2:
            self.xs = self.xs.unsqueeze(1)
            self.ys = self.ys.unsqueeze(1)
    
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
    
    
class sindy_loss(nn.CrossEntropyLoss):
    def __init__(self,
                 sindy_constructor: Callable,
                 sindy_update_rule_constructor: Callable,
                 sindy_library: ps.feature_library.base.BaseFeatureLibrary,
                 weight_rnn_pred: float = 1,
                 weight_sindy_pred: float = 1e-3,
                 weight_sindy_regularization: float = 1e-3,):
        super().__init__()
        
        self.sindy_constructor = sindy_constructor
        self.sindy_update_rule_constructor = sindy_update_rule_constructor
        self.sindy_library = sindy_library
        self.weight_rnn_pred = weight_rnn_pred
        self.weight_sindy_pred = weight_sindy_pred
        self.weight_regularization = weight_sindy_regularization
        
    def forward(self, action_prediction_rnn, target_prediction, control_inputs_sindy, q_prediction_rnn):    
        # control_inputs_sindy need to be a list of control inputs for the SINDy model with (prev_q_value, action, prev_action, reward)
        
        # create new instance of sindy model with sindy contstructor
        sindy_model = self.sindy_constructor()
        sindy_update_rule = self.sindy_update_rule_constructor(sindy_model)
        
        # compute the loss for:
        # - action prediction of RNN
        # - action prediction of SINDy model
        # - Q-Value prediction of SINDy model --> Only one sindy-prediction-loss needed; stay with Q-Values
        # - SINDy weight regularization
        
        # compute prediction error of RNN with nn.CrossEntropy
        loss_rnn_prediction = super(action_prediction_rnn, target_prediction)
        
        # compute Q-Value prediction error of SINDy model
        q_sindy = torch.zeros_like(q_prediction_rnn)
        for c in range(q_sindy.shape[-1]):
            q_sindy[:, :, c] = sindy_update_rule(*control_inputs_sindy)
        # use mse as metric
        loss_q_prediction = torch.nn.functional.mse_loss(q_sindy, q_prediction_rnn)
        
        # compute weight regularization of SINDy model
        loss_regularization_sindy_weights = torch.tensor(0.)
        for model in sindy_model:
            loss_regularization_sindy_weights += torch.sum(model.coefficients())
        
        return self.weight_rnn_pred * loss_rnn_prediction + self.weight_sindy_pred_x * loss_q_prediction + self.weight_regularization * loss_regularization_sindy_weights
        
    

def batch_train(
    model: BaseRNN,
    xs: torch.Tensor,
    ys: torch.Tensor,
    optimizer: torch.optim.Optimizer = None,
    loss_fn: nn.modules.loss._Loss = nn.CrossEntropyLoss(),
    ):

    """
    Trains a model with the given batch.
    """
    
    # predict y
    model.initial_state(batch_size=len(xs))
    y_pred = model(xs, model.get_state(detach=True), batch_first=True)[0]
    
    loss = 0
    for t in range(xs.shape[1]):
        loss += loss_fn(y_pred[:, t], ys[:, t])
    loss /= xs.shape[1]
    
    if loss.requires_grad:
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model, optimizer, loss
    

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
        
    if n_submodels > 1 and return_ensemble:
        model_backup = EnsembleRNN(models, voting_type=voting_type)
        optimizer_backup = optimizers
    else:
        model_backup = models[0]
        optimizer_backup = optimizers[0]
        
    return model_backup, optimizer_backup, loss