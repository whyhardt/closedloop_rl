import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, List, Union


class BaseRNN(nn.Module):
# class BaseRNN(torch.jit.ScriptModule):
    def __init__(
        self, 
        n_actions, 
        hidden_size, 
        init_value=0.5, 
        device=torch.device('cpu'),
        list_sindy_signals=['xQr', 'ca', 'cr'],
        ):
        super(BaseRNN, self).__init__()
        
        self.device = device
        
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size
               
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self._beta_init = 1
        
        # self._state = self.set_initial_state()
        
        # session history; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.history = {key: [] for key in list_sindy_signals}
        
        self.n_subnetworks = 0
    
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def set_initial_state(self, batch_size=1, return_dict=False):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
                
        for key in self.history.keys():
            self.history[key] = []
        
        # state dimensions: (habit_state, value_state, habit, value)
        # dimensions of states: (batch_size, submodel, substate, hidden_size)
        # submodel: one state per model in model ensemble
        # substate: one state per subnetwork in model
        self.set_state(
            torch.zeros([batch_size, 1, 3, self._hidden_size], dtype=torch.float, device=self.device),
            torch.zeros([batch_size, 1, 1, self._hidden_size], dtype=torch.float, device=self.device),
            torch.zeros([batch_size, 1, 2, self._hidden_size], dtype=torch.float, device=self.device),
            self.init_value + torch.zeros([batch_size, 1, self._n_actions], dtype=torch.float, device=self.device),
            torch.zeros([batch_size, 1, self._n_actions], dtype=torch.float, device=self.device),
            torch.zeros([batch_size, 1, self._n_actions], dtype=torch.float, device=self.device),
            )
        
        return self.get_state(return_dict=return_dict)
        
    def set_state(self, value_state: torch.Tensor, habit_state: torch.Tensor, uncertainty_state: torch.Tensor, value: torch.Tensor, habit: torch.Tensor, uncertainty: torch.Tensor):
        """this method sets the hidden state
        
        Args:
            state (Tuple[torch.Tensor]): hidden state
        """
        
        # self._state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
        self._state = (value_state, habit_state, uncertainty_state, value, habit, uncertainty)
      
    def get_state(self, detach=False, return_dict=False):
        """this method returns the hidden state
        
        Returns:
            Tuple[torch.Tensor]: tuple of habit state, value state, habit, value
        """
        
        state = self._state
        if detach:
            state = [s.detach() for s in state]

        if return_dict:
            keys = ['hidden_value', 'hidden_habit', 'hidden_uncertainty', 'value', 'habit', 'uncertainty']
            state = {keys[i]: state[i] for i in range(len(state))}
        
        return state
    
    def set_device(self, device: torch.device): 
        self.device = device
        
    def append_timestep_sample(self, key, old_value, new_value: Optional[torch.Tensor] = None, single_entries: bool = False):
        """appends a new timestep sample to the history. A timestep sample consists of the value at timestep t-1 and the value at timestep t

        Args:
            key (str): history key to which append the sample to
            old_value (_type_): value at timestep t-1 of shape (batch_size, feature_dim)
            new_value (_type_): value at timestep t of shape (batch_size, feature_dim)
        """
        
        # Do not append if model is in training mode for less overhead
        if hasattr(self, key) and getattr(self, key).training:
            return
        
        if new_value is None:
            new_value = torch.zeros_like(old_value) - 1
        
        old_value = old_value.view(-1, 1, old_value.shape[-1])
        new_value = new_value.view(-1, 1, new_value.shape[-1])
        sample = torch.cat([old_value, new_value], dim=1)
        if single_entries:
            # add each entry of the array as a single entry to the history. Useful to track e.g. latent variables
            for i in range(sample.shape[-1]):
                self.history[key + f'_{i}'].append(sample[:, :, i].view(-1, 2, 1))
        else:
            self.history[key].append(sample)
        
    def get_history(self, key):
        return self.history[key]
    
    def count_subnetworks(self):
        n_subnetworks = 0
        for name, module in self.named_modules():
            if name.startswith('x') and not '.' in name:
                n_subnetworks += 1
        return n_subnetworks
    
    def call_subnetwork(self, key, inputs, layer_hidden_state=3):
        if hasattr(self, key):
            # process input through different activations
            # Relu(input+bias) --> difficult with sindy
            # Sigmoid(input+bias) --> same
            # linear(input+bias) --> in SINDy: w*in + w*bias
            # Concat(Activations) or Sum(Activations)
            # pass to hidden layer
            # get hidden state (linear layer + activation + dropout)
            hidden_state = getattr(self, key)[0](inputs).swapaxes(1, 2)
            hidden_state = getattr(self, key)[1](hidden_state).swapaxes(1, 2)
            hidden_state = getattr(self, key)[2:layer_hidden_state](hidden_state)
            # get output variable (rest of subnetwork)
            output = getattr(self, key)[layer_hidden_state:](hidden_state)
            return output, hidden_state
        else:
            raise ValueError(f'Invalid key {key}.')
    
    def setup_subnetwork(self, input_size, hidden_size, dropout):
        seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            )
        
        for l in seq:
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_uniform_(l.weight)
        
        return seq
    
    def finetune_training(self, subnetwork: Union[nn.Sequential, str], keep_dropout: bool = False):
        # freeze all layers except for the last one
        if isinstance(subnetwork, str):
            if hasattr(self, subnetwork):
                subnetwork = getattr(self, subnetwork)
        
        for i in range(len(subnetwork)-1):
            # loop until len-1 to keep output layer trainable for finetuning
            if isinstance(subnetwork[i], nn.Linear):
                # freeze weights of linear layers
                for param in subnetwork[i].parameters():
                    param.requires_grad = False
            elif isinstance(subnetwork[i], nn.Dropout) and not keep_dropout:
                # disable dropout
                subnetwork[i].p = 0
            elif isinstance(subnetwork[i], nn.BatchNorm1d):
                # stop updating of running statistics in batch-norm layer
                for param in subnetwork[i].parameters():
                    param.requires_grad = False
                subnetwork[i].eval()

class RLRNN(BaseRNN):
    def __init__(
        self,
        n_actions, 
        hidden_size,
        init_value=0.5,
        list_sindy_signals=['xQf', 'xQr', 'xQc', 'xC', 'ca', 'cr'],
        dropout=0.,
        device=torch.device('cpu'),
        ):
        
        super(RLRNN, self).__init__(n_actions, hidden_size, init_value, device, list_sindy_signals)

        # define general network parameters
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._prev_action = torch.zeros(self._n_actions)
        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()
        self._shrink = nn.Tanhshrink()
        
        # trainable parameters
        self._beta_reward = nn.Parameter(torch.tensor(1.))
        self._beta_choice = nn.Parameter(torch.tensor(1.))
        # self._directed_exploration_bias_raw = nn.Parameter(torch.tensor(1.))
        
        # action-based subnetwork
        self.xC = self.setup_subnetwork(2, hidden_size, dropout)
        
        # action-based subnetwork for non-repeated action
        self.xCf = self.setup_subnetwork(1, hidden_size, dropout)
        
        # reward-blind subnetwork
        self.xQf = self.setup_subnetwork(1, hidden_size, dropout)
        
        # learning-rate subnetwork
        self.xLR = self.setup_subnetwork(2, hidden_size, dropout)
        
        # self.n_subnetworks = self.count_subnetworks()
        
        self._state = self.set_initial_state()
        
    def value_network(self, state: torch.Tensor, value: torch.Tensor, action: torch.Tensor, reward: torch.Tensor):
        """this method computes the reward-blind and reward-based updates for the Q-Values without considering the habit (e.g. last chosen action)
        
        Args:
            state (torch.Tensor): last hidden state
            value (torch.Tensor): last Q-Values
            action (torch.Tensor): chosen action (one-hot encoded)
            reward (torch.Tensor): received reward

        Returns:
            torch.Tensor: updated Q-Values
        """
        
        # add dimension to value, action and reward
        value = value.unsqueeze(-1)
        action = action.unsqueeze(-1)
        reward = reward.unsqueeze(-1).repeat((1, self._n_actions, 1))
        
        # get back previous states (same order as in return statement)
        state_chosen, learning_state, state_not_chosen = state[:, 0], state[:, 1], state[:, 2]
        next_value = torch.zeros_like(value) + value
            
        # reward sub-network for chosen action
        inputs = torch.concat([value, reward], dim=-1)
        learning_rate, _ = self.call_subnetwork('xLR', inputs)
        learning_rate = torch.nn.functional.sigmoid(learning_rate)
        rpe = reward - value
        update_chosen = learning_rate * rpe

        # reward sub-network for non-chosen action
        update_not_chosen, _ = self.call_subnetwork('xQf', value)
        update_not_chosen = self._shrink(update_not_chosen)
        
        next_value += update_chosen * action + update_not_chosen * (1-action)

        return next_value.squeeze(-1), learning_rate.squeeze(-1), torch.stack([state_chosen, learning_state, state_not_chosen], dim=1)
    
    def choice_network(self, state, value, action, repeated):
        
        # add dimension to value, action and repeated
        value = value.unsqueeze(-1)
        action = action.unsqueeze(-1)
        repeated = repeated.unsqueeze(-1).repeat((1, self._n_actions, 1))
        
        next_state = torch.zeros((state.shape[0], state.shape[-1]), device=self.device)
        next_value = torch.zeros_like(value)# + value
        
        # choice sub-network for chosen action
        inputs = torch.concat([value, repeated], dim=-1)
        update_chosen, _ = self.call_subnetwork('xC', inputs)
        update_chosen = self._tanh(self._shrink(update_chosen))
        
        # choice sub-network for non-chosen action
        update_not_chosen, _ = self.call_subnetwork('xCf', value)
        update_not_chosen = self._tanh(self._shrink(update_not_chosen))
        
        # next_state += state_update_chosen * action + state_update_not_chosen * (1-action)
        next_value += update_chosen * action + update_not_chosen * (1-action)
        
        return next_value.squeeze(-1), next_state.unsqueeze(1)
    
    # @torch.jit.script_method
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None, batch_first=False):
        """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
        Args:
            inputs (torch.Tensor): input tensor of form (seq_len, batch_size, n_actions + 1) or (batch_size, seq_len, n_actions + 1) if batch_first
            prev_state (Tuple[torch.Tensor]): tuple of previous state of form (habit state, value state, habit, value)

        Returns:
            torch.Tensor: updated Q-Values
            Tuple[torch.Tensor]: updated habit state, value state, habit, value
        """
        
        if len(inputs.shape) == 2:
            # unsqueeze time dimension
            if batch_first:
                inputs = inputs.unsqueeze(1)
            else:
                inputs = inputs.unsqueeze(0)
        
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        action = inputs[:, :, :-1].float()
        reward = inputs[:, :, -1].unsqueeze(-1).float()
        timesteps = torch.arange(inputs.shape[0])
        logits = torch.zeros(inputs.shape[0], inputs.shape[1], self._n_actions, device=self.device)
        
        # check if action is one-hot encoded
        action_oh = torch.zeros((inputs.shape[0], inputs.shape[1], self._n_actions), dtype=torch.float, device=self.device)
        if action.shape[-1] == 1:
            for i in range(inputs.shape[1]):
                action_oh[:, i, :] = torch.eye(self._n_actions, device=self.device)[action[:, i, 0].int()]
            action = action_oh
           
        if prev_state is not None:
            self.set_state(prev_state[0], prev_state[1], prev_state[2], prev_state[3], prev_state[4], prev_state[5])
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # get previous model state
        state = [s.squeeze(1) for s in self.get_state()]  # remove model dim for forward pass -> only one model
        reward_state, choice_state, uncertainty_state, reward_value, choice_value, uncertainty_value = state
        
        for t, a, r in zip(timesteps, action, reward):         
            # compute additional variables
            # compute whether action was repeated---across all batch samples
            repeated = 1*(torch.sum(torch.abs(a-self._prev_action), dim=-1) == 0).view(-1, 1)
            
            # compute the updates
            old_rv, old_cv, old_uv = reward_value.clone(), choice_value.clone(), uncertainty_value.clone() 
            reward_value, learning_rate, reward_state = self.value_network(reward_state, reward_value, a, r)
            choice_value, choice_state = self.choice_network(choice_state, choice_value, a, repeated)
            
            # logits[t, :, :] += (reward_value + action_value) * self.beta
            logits[t, :, :] += reward_value * self._beta_reward + choice_value * self._beta_choice
            
            self._prev_action = a.clone()
            
            # append timestep samples for SINDy
            self.append_timestep_sample('ca', a)
            self.append_timestep_sample('cr', r)
            self.append_timestep_sample('cp', 1-r)
            self.append_timestep_sample('ca_repeat', repeated)
            self.append_timestep_sample('cQ', old_rv)
            self.append_timestep_sample('xLR', torch.zeros_like(learning_rate), learning_rate)
            self.append_timestep_sample('xQf', old_rv, reward_value)
            self.append_timestep_sample('xC', old_cv, choice_value)
            self.append_timestep_sample('xCf', old_cv, choice_value)

        # add model dim again and set state
        self.set_state(reward_state.unsqueeze(1), choice_state.unsqueeze(1), uncertainty_state.unsqueeze(1), reward_value.unsqueeze(1), choice_value.unsqueeze(1), uncertainty_value.unsqueeze(1))
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            
        return logits, self.get_state()
    
    @property
    def beta(self):
        return torch.nn.functional.gelu(self._beta_reward)

    def set_initial_state(self, batch_size: int=1, return_dict=False):
        self._prev_action = torch.zeros(batch_size, self._n_actions, device=self.device)
        self._beta = self._beta_reward + torch.zeros([batch_size, 1], dtype=torch.float, device=self.device)
        return super().set_initial_state(batch_size, return_dict)

    
class EnsembleRNN:
    
    MEAN = 0
    MEDIAN = 1
    
    def __init__(self, model_list: List[BaseRNN], device=torch.device('cpu'), voting_type=0):
        self.device = device
        self.models = model_list
        self.voting_type = voting_type
        self.history = {key: [] for key in self.models[0].history.keys()}
        
    def __call__(self, inputs: torch.Tensor, prev_state: Optional[List[Tuple[torch.Tensor]]] = None, batch_first=False):
        if len(inputs.shape) == 2:
            # unsqueeze time dimension
            if batch_first:
                inputs = inputs.unsqueeze(1)
            else:
                inputs = inputs.unsqueeze(0)
            
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
            
        if prev_state is None:
            # initialize state if not provided
            self.initial_state(batch_size=inputs.shape[0])
            state = self.get_state()
        elif len(prev_state[0].shape) == 2:
            # repeat state for each model and return list of stacked states with shape (n_models, batch_size, hidden_size)
            self.set_state([prev_state for _ in range(len(self.models))])
            state = self.get_state()
        else:
            state = prev_state
        
        logits = []
        for inputs_t in inputs:
            history_call = {key: [] for key in self.models[0].history.keys()}
            logits_t = []
            states = []
            for i, model in enumerate(self.models):
                logits_i, state_i = model(inputs_t.unsqueeze(0), [s[:, i] for s in state])
                logits_t.append(logits_i)
                states.append(state_i)
                for key in history_call.keys():
                    if len(model.get_history(key)) > 0:
                        history_call[key].append(model.get_history(key)[-1])
            self.set_state(states)
            state = self.get_state()
            logits.append(self.vote(torch.stack(logits_t, dim=1), self.voting_type).squeeze(1))
            self.append_timestep_sample(history_call)

        logits = torch.concat(logits)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
        
        return logits, self.get_state()
    
    def __iter__(self):
        return iter(self.models)
    
    def __getitem__(self, index):
        return self.models[index]
    
    def __len__(self):
        return len(self.models)
    
    @staticmethod
    def vote(values: torch.Tensor, voting_type) -> torch.Tensor:
        if voting_type == 1:
            return torch.median(values, dim=1)[0].unsqueeze(1)
        elif voting_type == 0:
            return torch.mean(values, dim=1, keepdim=True)
        else:
            raise ValueError(f'Invalid ensemble type {voting_type}. Must be either 0 (mean) or 1 (median).')
    
    def get_voting_states(self):
        # get states that are voted for and 
        return [True if state_i.shape[-1] == self.models[0]._n_actions else False for state_i in self.models[0].get_state()]
    
    def set_state(self, states: List[Tuple[torch.Tensor]]):
        # vote for all non-hidden states i.e. visible values (habit and value) and copy all hidden states of each submodel
        # non-hidden states are identified by state[i].shape[-1] == self.models[0]._n_actions
        # transform all tuples to lists
        states = [list(s) for s in states]
        states_tensor = [None for _ in range(len(states[0]))]
        voting_states = self.get_voting_states()
        for i, voting in enumerate(voting_states):
            state_all_models = [s[i] for s in states]
            if voting:
                # if non-hidden state (aka visible states like habit and value) vote over all models and repeat the voted state for each model
                new_state = self.vote(torch.concatenate(state_all_models, dim=1), self.voting_type).repeat(1, len(self), 1)
            else:
                # if hidden state keep the state of each model
                new_state = torch.concatenate(state_all_models, dim=1)
            states_tensor[i] = new_state.clone()
        self._state = states_tensor
    
    def get_state(self, detach=False, return_dict=False):        
        state = self._state
        
        if detach:
            state = [s.detach() for s in state]
            
        if return_dict:
            keys = ['hidden_value', 'hidden_habit', 'hidden_uncertainty', 'value', 'habit', 'uncertainty']
            state = {keys[i]: state[i] for i in range(len(state))}
        else:
            state = tuple(state)
            
        return state
    
    def initial_state(self, batch_size=1, return_dict=False):
        self.history = {key: [] for key in self.models[0].history.keys()}
        state = []
        for model in self.models:
            state.append(model.set_initial_state(batch_size=batch_size))
        # state = list(zip(*state))
        # state = [torch.concat(s, dim=1) for s in state]
        # self._state = state
        self.set_state(state)
        return self.get_state(return_dict=return_dict)
    
    def get_history(self, key):
        if 'x' in key:
            # vote for internal values e.g. habit, reward-blind update, reward-based update
            return self.vote(torch.stack([model.get_history(key) for model in self.models]), self.voting_type)
        else:
            # control signals are equal for all models. It's sufficient to get only the first model's value
            return self.models[0].get_history(key)
        
    def append_timestep_sample(self, history_ensemble):
        for key in history_ensemble.keys():
            if 'x' in key:
                if len(history_ensemble[key]) > 0:
                    # vote history values
                    self.history[key].append(self.vote(torch.stack(history_ensemble[key], dim=1), self.voting_type).squeeze(1))
            elif 'c' in key:
                # control signals are equal for all models. It's sufficient to get only the first model's value
                self.history[key].append(history_ensemble[key][0])
            else:
                raise ValueError(f'Invalid history key {key}.')
        
    def set_device(self, device): 
        self.device = device
        
    def to(self, device: torch.device):
        for model in self.models:
            model = model.to(device)
            model.set_device(device)
        self.device = device
        return self
    
    def eval(self):
        for model in self.models:
            model.eval()
            
    def train(self):
        for model in self.models:
            model.train()