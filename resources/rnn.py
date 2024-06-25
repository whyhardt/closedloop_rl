import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy
from typing import Optional, Tuple, List, Union, Dict


class BaseRNN(nn.Module):
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
        
        self.prev_action = torch.zeros((1, n_actions), dtype=torch.float)
        
        # session history; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.history = {key: [] for key in list_sindy_signals}
    
    def forward(self, inputs, prev_state, batch_first=False):
        raise NotImplementedError('This method is not implemented.')
    
    def initial_state(self, batch_size=1, return_dict=False):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
        
        self.prev_action = torch.zeros((batch_size, self._n_actions), dtype=torch.float).to(self.device)
        
        for key in self.history.keys():
            self.history[key] = []
        
        self.set_state(
            torch.zeros([batch_size, 1, self._hidden_size], dtype=torch.float).to(self.device),
            torch.zeros([batch_size, 1, self._hidden_size], dtype=torch.float).to(self.device),
            torch.zeros([batch_size, 1, self._n_actions], dtype=torch.float).to(self.device),
            (self.init_value + torch.zeros([batch_size, 1, self._n_actions], dtype=torch.float)).to(self.device)
            )
        
        return self.get_state(return_dict=return_dict)
        
    def set_state(self, habit_state: torch.Tensor, value_state: torch.Tensor, habit: torch.Tensor, value: torch.Tensor):
        """this method sets the hidden state
        
        Args:
            state (Tuple[torch.Tensor]): hidden state
        """
        
        # self._state = dict(hidden_habit=habit_state, hidden_value=value_state, habit=habit, value=value)
        self._state = tuple([habit_state, value_state, habit, value])
      
    def get_state(self, detach=False, return_dict=False) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """this method returns the hidden state
        
        Returns:
            Tuple[torch.Tensor]: tuple of habit state, value state, habit, value
        """
        
        state = self._state
        if detach:
            state = [s.detach() for s in state]

        if return_dict:
            keys = ['hidden_habit', 'hidden_value', 'habit', 'value']
            state = {keys[i]: state[i] for i in range(len(state))}
        
        return state
    
    def set_device(self, device: torch.device): 
        self.device = device
        
    def append_timestep_sample(self, key, old_value, new_value: Optional[torch.Tensor] = None):
        """appends a new timestep sample to the history. A timestep sample consists of the value at timestep t-1 and the value at timestep t

        Args:
            key (str): history key to which append the sample to
            old_value (_type_): value at timestep t-1 of shape (batch_size, feature_dim)
            new_value (_type_): value at timestep t of shape (batch_size, feature_dim)
        """
        
        if new_value is None:
            new_value = torch.zeros(old_value.shape) - 1
        
        old_value = np.expand_dims(old_value.detach().cpu().numpy(), 1)
        new_value = np.expand_dims(new_value.detach().cpu().numpy(), 1)
        sample = np.concatenate([old_value, new_value], axis=1)
        self.history[key].append(sample)
        
    def get_history(self, key):
        return self.history[key]
    

class RLRNN(BaseRNN):
    def __init__(
        self,
        n_actions, 
        hidden_size,
        init_value=0.5,
        use_habit=False,
        last_output=False,
        last_state=False,
        list_sindy_signals=['xQf', 'xQr', 'xH', 'ca', 'ca[k-1]', 'cr'],
        device=torch.device('cpu'),
        ):
        
        super(RLRNN, self).__init__(n_actions, hidden_size, init_value, device, list_sindy_signals)
        
        # define level of recurrence
        self._vs, self._hs, self._vo, self._ho, self._wh = last_state, last_state, last_output, last_output, use_habit
        
        # define general network parameters
        self.init_value = init_value
        self._n_actions = n_actions
        # self.beta = nn.Parameter(torch.tensor(1., dtype=torch.float32, requires_grad=True))
        self._hidden_size = hidden_size
        
        # define input size according to arguments (network configuration)
        input_size = 2
        if self._vo:
            input_size += self._n_actions
        if self._vs:
            input_size += self._hidden_size
            
        # define layers
        # activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # habit subnetwork
        # make a list of modules
        self.habit_update = nn.Sequential(nn.Linear(1, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))
        
        # reward-blind subnetwork
        self.reward_blind_update = nn.Sequential(nn.Linear(n_actions-1, hidden_size), nn.Tanh(), nn.Linear(hidden_size, n_actions-1))
        
        # reward-based subnetwork
        self.hidden_layer_value = nn.Linear(input_size, hidden_size)
        self.reward_based_update = nn.Linear(hidden_size, 1)
        
    def value_network(self, state, value, action, reward):
        """this method computes the reward-blind and reward-based updates for the Q-Values without considering the habit (e.g. last chosen action)
        
        Args:
            state (torch.Tensor): last hidden state
            value (torch.Tensor): last Q-Values
            action (torch.Tensor): chosen action (one-hot encoded)
            reward (torch.Tensor): received reward

        Returns:
            torch.Tensor: updated Q-Values
        """

        # 1. reward-blind mechanism (forgetting) for all elements
        not_chosen_value = torch.sum((1-action) * value, dim=-1).view(-1, 1)
        blind_update = self.reward_blind_update(not_chosen_value)
        self.append_timestep_sample(key='xQf', old_value=value, new_value=action*value + (1-action)*blind_update)
        
        # 2. perseverance mechanism for previously chosen element
        # prev_chosen_value = torch.sum(self.prev_action * value, dim=-1).view(-1, 1)
        # habit_update = self.habit_update(prev_chosen_value)
        # self.append_timestep_sample('xH', value, self.prev_action*habit_update + (1-self.prev_action)*value)
        # self.append_timestep_sample('xH', value, habit_update + value)
        
        # 3. reward-based update for the chosen element        
        chosen_value = torch.sum(value * action, dim=-1).view(-1, 1)
        inputs = torch.cat([chosen_value, reward], dim=-1).float()
        
        if self._vo:
            inputs = torch.cat([inputs, value], dim=-1).float()
        if self._vs:
            inputs = torch.cat([inputs, state], dim=-1)
        
        next_state = self.tanh(self.hidden_layer_value(inputs))
        
        reward_update = self.reward_based_update(next_state)
        self.append_timestep_sample('xQr', value, action*reward_update + (1-action)*value)
        
        next_value = action * reward_update + (1-action) * blind_update# + self.prev_action * habit_update  

        return next_value, next_state
    
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor]] = None, batch_first=False):
        """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
        Args:
            inputs (torch.Tensor): input tensor of form (seq_len, batch_size, n_actions + 1) or (batch_size, seq_len, n_actions + 1) if batch_first
            prev_state (Tuple[torch.Tensor]): tuple of previous state of form (habit state, value state, habit, value)

        Returns:
            torch.Tensor: updated Q-Values
            Tuple[torch.Tensor]: updated habit state, value state, habit, value
        """
        
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)
        
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
                action_oh[:, i, :] = torch.eye(self._n_actions)[action[:, i, 0].int()]
            action = action_oh
            
        if prev_state is not None:
            self.set_state(*prev_state)
        else:
            self.initial_state(batch_size=inputs.shape[1], device=self.device)
        h_state, v_state, habit, value = self.get_state()
        # remove model dim for forward pass -> only one model
        h_state = h_state.squeeze(1)
        v_state = v_state.squeeze(1)
        habit = habit.squeeze(1)
        value = value.squeeze(1)
        
        for t, a, r in zip(timesteps, action, reward):
            self.append_timestep_sample('ca', a)
            self.append_timestep_sample('cr', r)
            self.append_timestep_sample('ca[k-1]', self.prev_action)
            
            # compute the updates
            value, v_state = self.value_network(v_state, value, a, r)
            
            self.prev_action = a
            
            logits[t, :, :] = value.clone()            
            # logits[t, :, :] = (self.sigmoid(logit)*self.beta).clone()
            
        # add model dim again and set state
        self.set_state(h_state.unsqueeze(1), v_state.unsqueeze(1), habit.unsqueeze(1), value.unsqueeze(1))
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            
        return logits, self.get_state()
    
    
class LSTM(BaseRNN):
    def __init__(
        self, 
        n_actions, 
        hidden_size, 
        init_value=0.5,
        device=torch.device('cpu'),
        ):
        super(LSTM, self).__init__(n_actions, hidden_size, init_value)
        
        self.device = device
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        
        input_size = n_actions + 1
            
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=False, num_layers=1)
        self.output_layer = nn.Linear(hidden_size, n_actions)
        
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor]] = None, batch_first=False):
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)
            
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        action = inputs[:, :, :-1].float()
        reward = inputs[:, :, -1].unsqueeze(-1).float()
                   
        # check if action is one-hot encoded
        if action.shape[-1] == 1:
            action = F.one_hot(action.squeeze(1).long(), num_classes=self._n_actions).float()
        
        if prev_state is not None:
            self.set_state(*prev_state)
        else:
            self.initial_state(batch_size=inputs.shape[1], device=self.device)
        c0, h0, _, value = self.get_state()
        
        # forward pass
        lstm_out, (c, h) = self.lstm(torch.concat((action, reward), dim=-1), (c0.swapaxes(0, 1), h0.swapaxes(0, 1)))
        logits = self.output_layer(lstm_out)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
        
        # add timestep samples
        self.append_timestep_sample('ca', action)
        self.append_timestep_sample('cr', reward)
        self.append_timestep_sample('xQr', value, logits[:, -1].view(-1, 1, self._n_actions))
        
        self.set_state(c.swapaxes(0, 1), h.swapaxes(0, 1), torch.zeros([inputs.shape[1], self._n_actions], dtype=torch.float).to(self.device), logits[:, -1].view(-1, 1, self._n_actions))
        
        return logits, self.get_state()
    
    
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
            inputs = inputs.unsqueeze(1)
            
        if prev_state is None:
            # initialize state if not provided
            prev_state = self.initial_state(batch_size=inputs.shape[0])
        if len(prev_state[0].shape) == 2:
            # repeat state for each model and return list of stacked states with shape (n_models, batch_size, hidden_size)
            self.set_state([prev_state for _ in range(len(self.models))])
            prev_state = self.get_state()
        
        logits = []
        states = []
        history_call = {key: [] for key in self.models[0].history.keys()}
        for i, model in enumerate(self.models):
            logits_i, state_i = model(inputs, [state[:, i] for state in prev_state], batch_first)
            logits.append(logits_i)
            states.append(state_i)
            for key in history_call.keys():
                if len(model.get_history(key)) > 0:
                    history_call[key].append(model.get_history(key))
        self.append_timestep_sample(history_call)

        # TODO: Check if voting logits and single states (xQf, xQr etc) leads to the same behavior
        logits = self.vote(torch.stack(logits, dim=1), self.voting_type).squeeze(1)
        states = self.set_state(states)
        
        return logits, states
    
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
        states = [list(state) for state in states]
        states_tensor = [None for _ in range(len(states[0]))]
        voting_states = self.get_voting_states()
        for i, voting in enumerate(voting_states):
            state = [state[i] for state in states]
            if voting:
                # if non-hidden state (aka visible states like habit and value) vote over all models and repeat the voted state for each model
                new_state = self.vote(torch.concatenate(state, dim=1), self.voting_type).repeat(1, len(self), 1)
            else:
                # if hidden state keep the state of each model
                new_state = torch.concatenate(state, dim=1)
            states_tensor[i] = new_state.clone()
        self._state = states_tensor
    
    def get_state(self, detach=False, return_dict=False):        
        state = self._state
        
        if detach:
            state = [s.detach() for s in state]
            
        if return_dict:
            keys = ['hidden_habit', 'hidden_value', 'habit', 'value']
            state = {keys[i]: state[i] for i in range(len(state))}
        else:
            state = tuple(state)
            
        return state
    
    def initial_state(self, batch_size=1, return_dict=False):
        self.history = {key: [] for key in self.models[0].history.keys()}
        state = []
        for model in self.models:
            state.append(model.initial_state(batch_size=batch_size))
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
                    self.history[key].append(self.vote(torch.tensor(np.stack(history_ensemble[key], axis=1)), self.voting_type).squeeze(1).numpy())
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