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

        self._beta_init = 1
        
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
        self._state = tuple([value_state, habit_state, uncertainty_state, value, habit, uncertainty])
      
    def get_state(self, detach=False, return_dict=False) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
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
            hidden_state = getattr(self, key)[:layer_hidden_state](inputs)
            # get output variable (rest of subnetwork)
            output = getattr(self, key)[layer_hidden_state:](hidden_state)
            mean, logvar = output[:, 0].view(-1, 1), output[:, 1].view(-1, 1)
            if self.training:
                z = self.reparameterization_trick(mean, logvar)
                return [z, mean, logvar], hidden_state
            else:
                return [mean, mean, logvar], hidden_state
        else:
            raise ValueError(f'Invalid key {key}.')
    
    def setup_subnetwork(self, input_size, hidden_size, dropout):
        seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
            )
        
        for l in seq:
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_uniform_(l.weight)
        
        return seq
    
    def reparameterization_trick(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z


class RLRNN(BaseRNN):
    def __init__(
        self,
        n_actions, 
        hidden_size,
        init_value=0.5,
        list_sindy_signals=['xQf', 'xQr', 'xQc', 'xH', 'ca', 'cr'],
        dropout=0.,
        device=torch.device('cpu'),
        ):
        
        super(RLRNN, self).__init__(n_actions, hidden_size, init_value, device, list_sindy_signals)

        # define general network parameters
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._prev_action = torch.zeros(self._n_actions)
        self._activation = nn.Sigmoid()
        self._shrink = nn.Tanhshrink()
        
        # trainable parameters
        self._beta_base = nn.Parameter(torch.tensor(1.))
        # self._directed_exploration_bias_raw = nn.Parameter(torch.tensor(1.))
        
        # action-based subnetwork
        self.xH = self.setup_subnetwork(2, hidden_size, dropout)
        
        # action-based subnetwork for non-repeated action
        self.xHf = self.setup_subnetwork(1, hidden_size, dropout)
        
        # # uncertainty subnetwork for chosen action
        # self.xU = self.setup_subnetwork(3, hidden_size, dropout)
        
        # # uncertainty subnetwork for not-chosen action
        # self.xUf = self.setup_subnetwork(1, hidden_size, dropout)
        
        # # beta subnetwork for undirected exploration
        # self.xB = self.setup_subnetwork(2, hidden_size, dropout)
        
        # reward-blind subnetwork
        self.xQf = self.setup_subnetwork(n_actions-1, hidden_size, dropout)
        
        # learning-rate subnetwork
        self.xLR = self.setup_subnetwork(2, hidden_size, dropout)
                
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
        
        # get back previous states (same order as in return statement)
        state_chosen, learning_state, state_not_chosen = state[:, 0], state[:, 1], state[:, 2]
        next_value = torch.zeros_like(value) + value
        eye = torch.eye(self._n_actions, device=self.device)
        
        vae_outputs = torch.zeros((value.shape[0], self._n_actions, 2))
        
        for i in range(self._n_actions):
            v = value[:, i].view(-1, 1)
            
            # reward sub-network for chosen action
            inputs = torch.concat([v, reward], dim=-1)
            learning_rate, learning_state = self.call_subnetwork('xLR', inputs)
            learning_rate[0] = torch.nn.functional.sigmoid(learning_rate[0])
            rpe = reward - v
            update_chosen = learning_rate[0] * rpe

            # reward sub-network for non-chosen action
            update_not_chosen, state_not_chosen = self.call_subnetwork('xQf', v)
            update_not_chosen[0] = self._shrink(update_not_chosen[0])
            
            next_value += eye[i] * (update_chosen * action[:, i].view(-1, 1) + update_not_chosen[0] * (1-action[:, i].view(-1, 1)))

            # store vae outputs (mean, logvar)
            vae_outputs[:, i, 0] = learning_rate[1].view(-1)
            vae_outputs[:, i, 0] = update_not_chosen[1].view(-1)
            vae_outputs[:, i, 1] = learning_rate[2].view(-1)
            vae_outputs[:, i, 1] = update_not_chosen[2].view(-1)
        
        return next_value, learning_rate[0], torch.stack([state_chosen, learning_state, state_not_chosen], dim=1), vae_outputs
    
    def action_network(self, state, value, action):
        
        next_state = torch.zeros((state.shape[0], state.shape[-1]), device=self.device)
        next_value = torch.zeros_like(value) + value
        eye = torch.eye(self._n_actions, device=self.device)
        
        vae_outputs = torch.zeros((value.shape[0], self._n_actions, 2))
        
        for i in range(self._n_actions):
            v = value[:, i].view(-1, 1)
            repeated = (action[:, i] * self._prev_action[:, i]).view(-1, 1)

            # action sub-network for chosen action
            inputs = torch.concat([v, repeated], dim=-1)
            update_chosen, state_update_chosen = self.call_subnetwork('xH', inputs)
            update_chosen[0] = self._shrink(update_chosen[0])
            
            # action sub-network for non-chosen action
            update_not_chosen, state_update_not_chosen = self.call_subnetwork('xHf', v)
            update_not_chosen[0] = self._shrink(update_not_chosen[0])
            
            next_state += state_update_chosen + state_update_not_chosen
            next_value += eye[i] * (update_chosen[0] * action[:, i].view(-1, 1) + update_not_chosen[0] * (1-action[:, i].view(-1, 1)))
            
            # store vae outputs (mean, logvar)
            vae_outputs[:, i, 0] = update_chosen[1].view(-1)
            vae_outputs[:, i, 0] = update_not_chosen[1].view(-1)
            vae_outputs[:, i, 1] = update_chosen[2].view(-1)
            vae_outputs[:, i, 1] = update_not_chosen[2].view(-1)

        # next_value = torch.nn.functional.tanh(next_value)
        self._prev_action = action
        
        return next_value, next_state.unsqueeze(1), vae_outputs
    
    def uncertainty_network(self, state, u_value, q_value, action, reward):
        
        uncertainty_state, beta_state = torch.zeros(state.shape[0], state.shape[-1], device=self.device), torch.zeros(state.shape[0], state.shape[-1], device=self.device)
        next_value = torch.zeros_like(u_value) + u_value
        next_value_mean = torch.zeros_like(u_value) + u_value
        eye = torch.eye(self._n_actions, device=self.device)
        
        # vae_outputs = torch.zeros((u_value.shape[0], self._n_actions+1, 2))
        vae_outputs = torch.zeros((u_value.shape[0], 1, 2))
        
        for i in range(self._n_actions):
            uv = u_value[:, i].view(-1, 1)
            qv = q_value[:, i].view(-1, 1)
            
            # uncertainty sub-network for chosen action
            inputs = torch.concat([uv, qv, reward], dim=-1)
            update_chosen, state_update = self.call_subnetwork('xU', inputs)
            update_chosen[0] = self._shrink(update_chosen[0])
            uncertainty_state += state_update
            
            # uncertainty sub-network for non-chosen action
            update_not_chosen, state_update = self.call_subnetwork('xUf', uv)
            update_not_chosen[0] = self._shrink(update_not_chosen[0])
            uncertainty_state += state_update
            
            next_value += eye[i] * (update_chosen[0] * action[:, i].view(-1, 1) + update_not_chosen[0] * (1-action[:, i].view(-1, 1)))
            next_value_mean += eye[i] * (update_chosen[1] * action[:, i].view(-1, 1) + update_not_chosen[1] * (1-action[:, i].view(-1, 1)))
            
            # store vae outputs (mean, logvar)
            # vae_outputs[:, i, 0] = update_chosen[1].view(-1)
            # vae_outputs[:, i, 0] = update_not_chosen[1].view(-1)
            # vae_outputs[:, i, 1] = update_chosen[2].view(-1)
            # vae_outputs[:, i, 1] = update_not_chosen[2].view(-1)
        
        # next_value = torch.nn.functional.tanh(next_value)
        # next_value_mean = torch.nn.functional.tanh(next_value_mean)
        
        # undirected/random exploration
        inputs = torch.concat([next_value_mean, self._beta], dim=-1)
        update_beta, beta_state = self.call_subnetwork('xB', next_value_mean)
        update_beta[0] = self._shrink(update_beta[0])
        self._beta = self._beta_base + update_beta[0]
        self._beta = torch.nn.functional.relu(self._beta)
        # update_beta[0] = torch.nn.functional.relu(update_beta[0])
        # self._beta = update_beta[0]
        
        # store vae outputs (mean, logvar)
        vae_outputs[:, 0, 0] = update_beta[1].view(-1)
        vae_outputs[:, 0, 1] = update_beta[2].view(-1)
        
        return next_value, torch.stack([uncertainty_state, beta_state], dim=1), vae_outputs
    
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
        vae_outputs = torch.zeros(inputs.shape[0], inputs.shape[1], 2+2, 2, device=self.device)
        
        # check if action is one-hot encoded
        action_oh = torch.zeros((inputs.shape[0], inputs.shape[1], self._n_actions), dtype=torch.float, device=self.device)
        if action.shape[-1] == 1:
            for i in range(inputs.shape[1]):
                action_oh[:, i, :] = torch.eye(self._n_actions, device=self.device)[action[:, i, 0].int()]
            action = action_oh
            
        if prev_state is not None:
            self.set_state(*prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # get previous model state
        state = [s.squeeze(1) for s in self.get_state()]  # remove model dim for forward pass -> only one model
        state_reward, state_action, state_uncertainty, value_reward, value_action, value_uncertainty = state
        
        for t, a, r in zip(timesteps, action, reward):         
            
            # compute the updates
            old_rv, old_av, old_uv = value_reward.clone(), value_action.clone(), value_uncertainty.clone() 
            value_reward, learning_rate, state_reward, vae_outputs_reward = self.value_network(state_reward, value_reward, a, r)
            value_action, state_action, vae_outputs_action = self.action_network(state_action, value_action, a)
            # value_uncertainty, state_uncertainty, vae_outputs_uncertainty = self.uncertainty_network(state_uncertainty, value_uncertainty, value_reward, a, r)
            
            value_reward = torch.clip(value_reward, 0, 1)
            # action_value = torch.clip(action_value, -1, 1)
            # uncertainty_value = torch.clip(uncertainty_value, -1, 1)
            
            # logits[t, :, :] += (value_reward + value_action + value_uncertainty) * self._beta
            logits[t, :, :] += (value_reward + value_action) * self._beta
            # logits[t, :, :] += value_reward * self._beta + value_action
            
            # vae_outputs[t] = torch.concat((vae_outputs_reward, vae_outputs_action, vae_outputs_uncertainty), dim=-2)
            vae_outputs[t] = torch.concat((vae_outputs_reward, vae_outputs_action), dim=-2)
            
            # append timestep samples for SINDy
            self.append_timestep_sample('ca', a)
            self.append_timestep_sample('cr', r)
            self.append_timestep_sample('cp', 1-r)
            self.append_timestep_sample('ca_repeat', 1-torch.sum(torch.abs(a-self._prev_action), dim=-1)/self._n_actions)
            self.append_timestep_sample('cQ', old_rv)
            # self.append_timestep_sample('cU', value_uncertainty, single_entries=True)
            self.append_timestep_sample('xLR', torch.zeros_like(learning_rate), learning_rate)
            self.append_timestep_sample('xQf', old_rv, a*old_rv + (1-a)*value_reward)
            self.append_timestep_sample('xH', old_av, (1-a)*old_av + a*value_action)
            self.append_timestep_sample('xHf', old_av, a*old_av + (1-a)*value_action)
            # self.append_timestep_sample('xU', old_uv, (1-a)*old_uv + a*value_uncertainty)
            # self.append_timestep_sample('xUf', old_uv, a*old_uv + (1-a)*value_uncertainty)
            # self.append_timestep_sample('xB', torch.zeros_like(self._beta), self._beta - self._beta_base)

        # add model dim again and set state
        self.set_state(state_reward.unsqueeze(1), state_action.unsqueeze(1), state_uncertainty.unsqueeze(1), value_reward.unsqueeze(1), value_action.unsqueeze(1), value_uncertainty.unsqueeze(1))
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            vae_outputs = vae_outputs.permute(1, 0, 2, 3)
            
        return logits, self.get_state(), vae_outputs

    def set_initial_state(self, batch_size=1, return_dict=False):
        # self._prev_update = torch.zeros((batch_size, 2, self._n_actions), dtype=torch.float).to(self.device)
        self._prev_action = torch.zeros(batch_size, self._n_actions, device=self.device)
        self._beta = self._beta_base + torch.zeros([batch_size, 1], dtype=torch.float, device=self.device)
        return super().set_initial_state(batch_size, return_dict)
    
    # @property
    # def _directed_exploration_bias(self):
    #     # return torch.nn.functional.sigmoid(self._directed_exploration_bias_raw)
    #     # return torch.tensor(0., device=self.device)
    #     return self._directed_exploration_bias_raw
      
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
            self.set_initial_state(batch_size=inputs.shape[1])
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