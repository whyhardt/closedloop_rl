import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict


class ZerosModule(nn.Module):
    
    def __init__(self, embedding_dim):
        super(ZerosModule, self).__init__()
        
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor):
        return torch.zeros((x.shape[-1], self.embedding_dim), device=x.device)


class BaseRNN(nn.Module):
    
    module_types = {
        'param': 0,
        'linear': 1,
        'rnn': 2,
    }
    
    def __init__(
        self, 
        n_actions: int, 
        hidden_size: int,
        n_participants: int,
        signals,
        module_inputs,
        module_filter,
        dropout = 0.0,
        ):
        # signals starting with: 'x_' -> rnn modules; '_p' -> linear modules (linear+activation); '_c' -> observed/derived control signals
        
        super(BaseRNN, self).__init__()
        
        self.device = torch.device('cpu')
        
        self._signals = signals
        self._module_inputs = module_inputs
        self._module_filter = module_filter
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._n_participants = n_participants
        self._sigmoid = nn.Sigmoid()
        
        # initialize RNN
        self.init_modules()
        # recorded signals across all trials; used for sindy training; training variables start with 'x' and control parameters with 'c' 
        self.recordings = {key: [] for key in signals}
        
        embedding_dim = 8
        if n_participants > 1:
            self.participant_embedding = nn.Embedding(n_participants, embedding_dim)
        else:
            self.participant_embedding = ZerosModule(embedding_dim)
        
        # here starts the custom RNN setup
        for signal in signals:
            if 'x_' in signal:
                self.rnn_modules[signal] = self.setup_module(input_size=2+embedding_dim, hidden_size=hidden_size, dropout=dropout, module_type=self.module_types['rnn'])
                self.rnn_modules_types[signal] = self.module_types['rnn']
            elif 'p_' in signal:
                self.rnn_modules[signal] = self.setup_module(input_size=embedding_dim, hidden_size=hidden_size, activation=nn.ReLU, module_type=self.module_types['linear'])
                self.rnn_modules_types[signal] = self.module_types['linear']
                
        self.set_initial_state()
        
    def forward(self, inputs: Dict[str, torch.Tensor], prev_state: Optional[Dict[str, torch.Tensor]] = None, batch_first=False) -> Dict[str, torch.Tensor]:
        """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
        Args:
            inputs (torch.Tensor): input tensor of form (seq_len, batch_size, n_actions + 1) or (batch_size, seq_len, n_actions + 1) if batch_first
            prev_state (Tuple[torch.Tensor]): tuple of previous state of form (habit state, value state, habit, value)

        Returns:
            torch.Tensor: updated Q-Values
            Tuple[torch.Tensor]: updated habit state, value state, habit, value
        """
        
        for control_signal in inputs:
            # check that all input keys appear in list_control_signals
            if not control_signal in self._signals:
                raise ValueError(f'Input control signal {control_signal} is not in the list valid signals ({self._signals}).')
        
            if batch_first:
                inputs[control_signal] = inputs[control_signal].swapaxes(1, 0)
        
        # set model state if provided
        if prev_state is not None:
            self.set_state(prev_state)
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # compute participant embedding if available
        if hasattr(self, 'participant_embedding'):
            inputs['c_ParticipantID'] = self.participant_embedding(inputs['c_ParticipantID'].int()).repeat((1, 1, self._n_actions, 1))
        else:
            inputs['c_ParticipantID'] = torch.zeros((inputs['c_Action'].shape[:-1], self._n_actions, 1), device=self.device)
        
        timestep_array = torch.arange(inputs['c_Action'].shape[0])
        logits_array = torch.zeros_like(inputs['c_Action'], device=self.device)
        for timestep in timestep_array:
            
            # extract control signal at current timestep 
            inputs_t = {control_signal: inputs[control_signal][timestep] for control_signal in inputs}
            
            # perform custom update
            logits, next_state, inputs_t = self.update(self.get_state(), inputs_t)
            
            # compute the updates
            self.logits = logits.clone()
            logits_array[timestep, :, :] += logits
            
            # record states and control signals
            for state in next_state:
                self.recording(state, self.get_state()[state], next_state[state])
            # record control signals
            for control_signal in inputs_t:
                self.recording(control_signal, inputs_t[control_signal])
            
            # save current state
            self.set_state(next_state)
            
        if batch_first:
            logits_array = logits_array.swapaxes(1, 0)
            for control_signal in inputs:
                inputs[control_signal] = inputs[control_signal].swapaxes(1, 0)
            
        return {'y_Action': logits_array}
    
    def update(self, state: dict, inputs: dict):
        # update is unique for each RNN and should therefore overwrite the method given by the parent class
        # all other standard computations are handled in the forward method, which is equal for all child classes 
        
        # initialize dict for updates
        updates = {}
        next_state = {}
        
        # add here custom input derivation code e.g. action repetition and add to inputs
        
        # here starts the custom update code
        # get update for chosen, reward-based value
        key = 'x_ValueRewardCH'
        module_inputs = state[key].unsqueeze(-1)
        for input_signal in self._module_inputs[key]:
            module_inputs = torch.concat((module_inputs, inputs[input_signal].unsqueeze(-1)), dim=-1)
        module_inputs = torch.concat((module_inputs, inputs['c_ParticipantID']), dim=-1)
        # TODO: use datafilter_setup like for sindy
        updates[key] = self.call_module(key, module_inputs)[:, :, 0]
        for module_filter in self._module_filter:
            filter_signal = self._module_filter[module_filter][0]
            filter_value = self._module_filter[module_filter][1]
            updates[key] = updates[key] * (inputs[filter_signal] == filter_value)
        
        # set update for every rnn module
        next_state[key] = self._sigmoid(state[key] + updates[key])
        
        # perform logit computations if necessary
        logits = next_state['x_ValueRewardCH']
        
        # return logits
        return logits, next_state, inputs
    
    def init_modules(self):
        self.rnn_modules = nn.ModuleDict()
        self.rnn_modules_types = {}
    
    def reward_prediction_error(self, value, reward):
        return reward - value
    
    def set_initial_state(self, batch_size: int = 1):
        """this method initializes the hidden state. For BaseRNN there's only one state.
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
        
        # set a state of 0 for each rnn module in self.rnn_modules
        state = {}
        for module in self.rnn_modules:
            if 'x_' in module:
                state[module] = torch.zeros((batch_size, self._n_actions), dtype=torch.float32)
        
        for key in self.recordings.keys():
            self.recordings[key] = []        
        
        self.set_state(state)
                
    def set_state(self, state: dict, batch_size: Optional[int] = 1):
        """this method sets the hidden state
        
        Args:
            state (Tuple[torch.Tensor]): hidden state
        """
        
        for s in state:
            if len(state[s].shape) == 1:
                state[s] = state[s].unsqueeze(0)
            if state[s].shape[0] == 1 and batch_size > 1:
                state[s] = state[s].repeat(0, batch_size)
        
        self._state = state
      
    def get_state(self, detach=False):
        """this method returns the hidden state
        
        Returns:
            Tuple[torch.Tensor]: tuple of habit state, value state, habit, value
        """
        
        state = self._state
        if detach:
            for s in state:
                state[s] = state[s].detach()

        return state
        
    def recording(self, key, old_value, new_value: Optional[torch.Tensor] = None):
        """appends a new timestep sample to the history. A timestep sample consists of the value at timestep t-1 and the value at timestep t

        Args:
            key (str): history key to which append the sample to
            old_value (_type_): value at timestep t-1 of shape (batch_size, feature_dim)
            new_value (_type_): value at timestep t of shape (batch_size, feature_dim)
        """
        
        # Do not append if model is in training mode for less overhead
        if key in self.rnn_modules and self.rnn_modules[key].training:
            return
        
        if new_value is None:
            new_value = torch.zeros_like(old_value) - 1
        
        sample = torch.stack([old_value, new_value], dim=1)
        self.recordings[key].append(sample)
        
    def get_recording(self, key):
        return self.recordings[key]
    
    def setup_module(self, input_size:int=None, hidden_size:int=None, dropout:float=None, activation:nn.Module=nn.Identity, module_type:int=None):
        if module_type == 0:
            module = nn.Parameter(torch.tensor(1.))
        elif module_type == 1:
            module = nn.Sequential(
                nn.Linear(input_size, 1),
                activation(),
            )
        elif module_type==2:
            module = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                )
        
            for l in module:
                if isinstance(l, nn.Linear):
                    torch.nn.init.xavier_normal_(l.weight)
                    # torch.nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
        
        return module
    
    def call_module(self, key, inputs):
        if key in self.rnn_modules:
            if isinstance(self.rnn_modules[key], nn.Sequential):
                output = self.rnn_modules[key](inputs)
            else:
                output = self.rnn_modules[key] * inputs 
        else:
            raise ValueError(f'Invalid key {key}.')
        
        # TODO: record the values and updates
        
        return output
    
    def to(self, device: torch.device):
        self.device = device
        for state in self._state:
            self._state = self._state[state].to(device)
        return super(BaseRNN, self).to(device)


class RLRNN(BaseRNN):
    def __init__(
        self,
        n_actions: int, 
        hidden_size: int,
        n_participants: int,
        signals = ['x_ValueRewardCH_LR', 'x_ValueRewardCH', 'x_ValueRewardNC', 'x_ValueChoiceCH', 'x_ValueChoiceNC', 'c_Action', 'c_Reward', 'c_ParticipantID', 'c_ActionRepeated'],
        dropout = 0.,
        counterfactual = False,
        device = torch.device('cpu'),
        ):
        
        super(RLRNN, self).__init__(n_actions=n_actions, hidden_size=hidden_size, device=device, signals=signals)

        # additional general network attributes
        self._prev_action = torch.zeros(self._n_actions)
        self._counterfactual = counterfactual
        
        # participant-embedding layer
        if n_participants > 0:
            emb_size = 8
            self.participant_embedding = nn.Embedding(n_participants, emb_size)
            self._beta_reward = self.setup_module(input_size=emb_size, activation=nn.ReLU(), module_type=self.module_types['linear'])
            self._beta_choice = self.setup_module(input_size=emb_size, activation=nn.ReLU(), module_type=self.module_types['linear'])
        else:
            emb_size = 0
            self._beta_reward = self.setup_module(module_type=self.module_types['param'])
            self._beta_choice = self.setup_module(module_type=self.module_types['param'])
        
        # update modules for choice-based values (chosen and not-chosen)
        self.rnn_modules['x_C'] = self.setup_module(2+emb_size, hidden_size, dropout, module_type=self.module_types['rnn'])
        self.rnn_modules['x_C_nc'] = self.setup_module(1+emb_size, hidden_size, dropout, module_type=self.module_types['rnn'])
        
        # update modules for reward-based values (chosen and not-chosen)
        # in case of chosen: computation of learning rate 
        self.rnn_modules['x_V_LR'] = self.setup_module(2+int(counterfactual)+emb_size, hidden_size, dropout, module_type=self.module_types['rnn'])
        self.rnn_modules['x_V_nc'] = self.setup_module(1+emb_size, hidden_size, dropout, module_type=self.module_types['rnn'])
        
        # initialize RNN
        self.set_initial_state()
    
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor]] = None, batch_first=False):
        """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
        Args:
            inputs (torch.Tensor): input tensor of form (seq_len, batch_size, n_actions + 1) or (batch_size, seq_len, n_actions + 1) if batch_first
            prev_state (Tuple[torch.Tensor]): tuple of previous state of form (habit state, value state, habit, value)

        Returns:
            torch.Tensor: updated Q-Values
            Tuple[torch.Tensor]: updated habit state, value state, habit, value
        """
        
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        # extract observed inputs (action, reward, participant_id) 
        action_array = inputs[:, :, :self._n_actions].float()
        reward_array = inputs[:, :, self._n_actions:2*self._n_actions].float()
        participant_id_array = inputs[:, :, -1].unsqueeze(-1).int()
        
        # set model state if provided
        if prev_state is not None:
            self.set_state(prev_state[0], prev_state[1], prev_state[2], prev_state[3])
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # get previous model state
        value_reward, value_choice = self.get_state()  # remove model dim for forward pass -> only one model
        
        # get participant embedding
        if self._n_participants > 0:
            participant_embedding_array = self.participant_embedding(participant_id_array).repeat(1, 1, self._n_actions, 1)
            beta_reward = self._beta_reward(participant_embedding_array[0, :, 0])
            beta_choice = self._beta_choice(participant_embedding_array[0, :, 0])
        else:
            participant_embedding_array = torch.zeros((*inputs.shape[:-1], 2, 0), device=self.device)
            beta_reward = self._beta_reward
            beta_choice = self._beta_choice
        
        timestep_array = torch.arange(inputs.shape[0])
        logits_array = torch.zeros_like(action_array, device=self.device)
        for timestep, action, reward, participant in zip(timestep_array, action_array, reward_array, participant_embedding_array):         
            
            # compute additional variables
            # compute whether action was repeated---across all batch samples
            repeated = 1*(torch.sum(torch.abs(action-self._prev_action), dim=-1) == 0).view(-1, 1)
            
            # compute the updates
            value_reward_prev, value_choice_prev = value_reward.clone(), value_choice.clone() 
            value_reward, learning_rate = self.value_network(value_reward, action, reward, participant)
            value_choice = self.choice_network(value_choice, action, repeated, participant)
            
            logits_array[timestep, :, :] += value_reward * beta_reward + value_choice * beta_choice
            
            self._prev_action = action.clone()
            
            # signal tracking to collect training samples for SINDy
            if not torch.is_grad_enabled():
                # control signals
                self.recording('c_a', action)  # action
                self.recording('c_r', reward[:, :1] if not self._counterfactual else reward[:, action.argmax(dim=-1)])  # reward chosen
                self.recording('c_r_cf', reward[:, 1:] if not self._counterfactual else reward[:, (1-action).argmax(dim=-1)])  # reward not-chosen (aka counterfactual)
                self.recording('c_a_repeat', repeated)  # action repeated
                self.recording('c_V', value_reward_prev)  # reward-based values
                
                # dynamical variabels
                self.recording('x_V_LR', torch.zeros_like(learning_rate), learning_rate)
                self.recording('x_V_LR_cf', torch.zeros_like(learning_rate), learning_rate if reward[:, -1].mean() != -1 else torch.zeros_like(learning_rate))
                self.recording('x_V_nc', value_reward_prev, value_reward)
                self.recording('x_C', value_choice_prev, value_choice)
                self.recording('x_C_nc', value_choice_prev, value_choice)

        # add model dim again and set state
        self.set_state(value_reward, value_choice)
        
        if batch_first:
            logits_array = logits_array.permute(1, 0, 2)
            
        return logits_array, self.get_state()
        
    def value_network(self, value: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, participant_embedding: torch.Tensor):
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
        
        counterfactual = True
        reward = reward.unsqueeze(-1)
        if reward[:, 1].mean() == -1:
            counterfactual = False
            reward = reward[:, 0].unsqueeze(1).repeat((1, self._n_actions, 1))

        next_value = torch.zeros_like(value) + value
        
        # learning rate sub-network for chosen and not-chosen action
        if self.x_V_LR[0].in_features-participant_embedding.shape[-1]-2 > 0:
            inputs = torch.concat([value, reward, action, participant_embedding], dim=-1)
        else:
            inputs = torch.concat([value, reward, participant_embedding], dim=-1)
        learning_rate, _ = self.call_module('x_V_LR', inputs)
        learning_rate = self._sigmoid(learning_rate)
        rpe = reward - value
        update_chosen = learning_rate * rpe
        
        if counterfactual:
            # counterfactual feedback
            # update_not_chosen given via counterfactual learning
            
            # apply update_chosen for all values and don't apply forgetting since participant sees all rewards at every trial
            next_value += update_chosen
        else:
            # no counterfactual feedback -> apply update for not-chosen values
            
            # reward sub-network for non-chosen action
            inputs = torch.concat([value, participant_embedding], dim=-1)
            update_not_chosen, _ = self.call_module('x_V_nc', inputs)
            
            # apply update_chosen only for chosen option and update_not_chosen for not-chosen option
            next_value += update_chosen * action + update_not_chosen * (1-action)
            
        return next_value.squeeze(-1), learning_rate.squeeze(-1)
    
    def choice_network(self, value: torch.Tensor, action: torch.Tensor, repeated: torch.Tensor, participant_embedding: torch.Tensor):
        
        # add dimension to value, action and repeated
        value = value.unsqueeze(-1)
        action = action.unsqueeze(-1)
        repeated = repeated.unsqueeze(-1).repeat((1, self._n_actions, 1))
        
        next_value = torch.zeros_like(value) + value
        
        # choice sub-network for chosen action
        inputs = torch.concat([value, repeated, participant_embedding], dim=-1)
        update_chosen, _ = self.call_module('x_C', inputs)
        
        # choice sub-network for non-chosen action
        inputs = torch.concat([value, participant_embedding], dim=-1)
        update_not_chosen, _ = self.call_module('x_C_nc', inputs)
        
        # next_state += state_update_chosen * action + state_update_not_chosen * (1-action)
        next_value += update_chosen * action + update_not_chosen * (1-action)
        next_value = self._sigmoid(next_value)
        
        return next_value.squeeze(-1)
    
    def set_initial_state(self, batch_size: int = 1):
        
        # additional steps to initialize RL-RNN
        self._prev_action = torch.zeros_like((batch_size, self._n_actions), device=self.device)
        
        # standard steps to initialize BaseRNN
        super().set_initial_state(batch_size=batch_size)
