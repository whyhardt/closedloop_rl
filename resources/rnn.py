import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class BaseRNN(nn.Module):
# class BaseRNN(torch.jit.ScriptModule):
    def __init__(
        self, 
        n_actions, 
        hidden_size, 
        init_value=0.5, 
        device=torch.device('cpu'),
        list_sindy_signals=['x_V', 'c_a', 'c_r'],
        ):
        super(BaseRNN, self).__init__()
        
        self.device = device
        
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size

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
            keys = ['x_V_state', 'x_C_state', 'x_U_state', 'x_V', 'x_C', 'x_U']
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
    
    def call_subnetwork(self, key, inputs, layer_hidden_state=2):
        if hasattr(self, key):
            # hidden_state = getattr(self, key)[0](inputs).swapaxes(1, 2)
            # hidden_state = getattr(self, key)[1](hidden_state).swapaxes(1, 2)
            # hidden_state = getattr(self, key)[2:layer_hidden_state](hidden_state)
            hidden_state = getattr(self, key)[:layer_hidden_state](inputs)
            # get output variable (rest of subnetwork)
            output = getattr(self, key)[layer_hidden_state:](hidden_state)
            return output, hidden_state
        else:
            raise ValueError(f'Invalid key {key}.')
    
    def setup_subnetwork(self, input_size, hidden_size, dropout):
        seq = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            )
        
        for l in seq:
            if isinstance(l, nn.Linear):
                torch.nn.init.xavier_normal_(l.weight)
                # torch.nn.init.kaiming_normal_(l.weight, nonlinearity='relu')
        
        return seq
    

class RLRNN(BaseRNN):
    def __init__(
        self,
        n_actions:int, 
        hidden_size:int,
        n_participants:int=0,
        init_value=0.5,
        list_sindy_signals=['x_V', 'x_V_nc', 'x_C', 'x_C_nc', 'c_a', 'c_r'],
        dropout=0.,
        counterfactual=False,
        device=torch.device('cpu'),
        ):
        
        super(RLRNN, self).__init__(n_actions, hidden_size, init_value, device, list_sindy_signals)

        # define general network parameters
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size
        self._prev_action = torch.zeros(self._n_actions)
        self._relu = nn.ReLU()
        self._sigmoid = nn.Sigmoid()
        self._n_participants = n_participants
        self._counterfactual = counterfactual
        
        # participant-embedding layer
        if n_participants > 0:
            emb_size = 8
            self.participant_embedding = nn.Embedding(n_participants, emb_size)
            self._beta_reward = nn.Sequential(nn.Linear(emb_size, 1), nn.ReLU())
            self._beta_choice = nn.Sequential(nn.Linear(emb_size, 1), nn.ReLU())
        else:
            emb_size = 0
            self._beta_reward = nn.Parameter(torch.tensor(1.))
            self._beta_choice = nn.Parameter(torch.tensor(1.))
        
        # action-based subnetwork
        self.x_C = self.setup_subnetwork(2+emb_size, hidden_size, dropout)
        
        # action-based subnetwork for non-repeated action
        self.x_C_nc = self.setup_subnetwork(1+emb_size, hidden_size, dropout)
        
        # reward-based learning-rate subnetwork for chosen action (and counterfactual if applies)
        # add_action_dim = 1 if counterfactual else 0
        self.x_V_LR = self.setup_subnetwork(2+int(counterfactual)+emb_size, hidden_size, dropout)
        
        # reward-based subnetwork for not-chosen action
        self.x_V_nc = self.setup_subnetwork(1+emb_size, hidden_size, dropout)
        
        # self.n_subnetworks = self.count_subnetworks()
        
        self._state = self.set_initial_state()
        
    def value_network(self, state: torch.Tensor, value: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, participant_embedding: torch.Tensor):
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

        # get back previous states (same order as in return statement)
        state_chosen, learning_state, state_not_chosen = state[:, 0], state[:, 1], state[:, 2]
        next_value = torch.zeros_like(value) + value
        
        # learning rate sub-network for chosen and not-chosen action
        if self.x_V_LR[0].in_features-participant_embedding.shape[-1]-2 > 0:
            inputs = torch.concat([value, reward, action, participant_embedding], dim=-1)
        else:
            inputs = torch.concat([value, reward, participant_embedding], dim=-1)
        learning_rate, _ = self.call_subnetwork('x_V_LR', inputs)
        learning_rate = self._sigmoid(learning_rate)
        rpe = reward - value
        update_chosen = learning_rate * rpe
        # update_chosen += value
        
        if counterfactual:
            # counterfactual feedback
            # update_not_chosen given via counterfactual learning
            
            # apply update_chosen for all values and don't apply forgetting since participant sees all rewards at every trial
            next_value += update_chosen
        else:
            # no counterfactual feedback -> apply update for not-chosen values
            
            # reward sub-network for non-chosen action
            inputs = torch.concat([value, participant_embedding], dim=-1)
            update_not_chosen, _ = self.call_subnetwork('x_V_nc', inputs)
            
            # apply update_chosen only for chosen option and update_not_chosen for not-chosen option
            # sigmoid applied only to not-chosen action because chosen action is already bounded to range [0, 1]
            # next_value += update_chosen * action + self._sigmoid(value + update_not_chosen) * (1-action)
            next_value += update_chosen * action + update_not_chosen * (1-action)
            
        return next_value.squeeze(-1), learning_rate.squeeze(-1), torch.stack([state_chosen, learning_state, state_not_chosen], dim=1)
    
    def choice_network(self, state: torch.Tensor, value: torch.Tensor, action: torch.Tensor, repeated: torch.Tensor, participant_embedding: torch.Tensor):
        
        # add dimension to value, action and repeated
        value = value.unsqueeze(-1)
        action = action.unsqueeze(-1)
        repeated = repeated.unsqueeze(-1).repeat((1, self._n_actions, 1))
        
        next_state = torch.zeros((state.shape[0], state.shape[-1]), device=self.device)
        next_value = torch.zeros_like(value) + value
        
        # choice sub-network for chosen action
        inputs = torch.concat([value, repeated, participant_embedding], dim=-1)
        update_chosen, _ = self.call_subnetwork('x_C', inputs)
        
        # choice sub-network for non-chosen action
        inputs = torch.concat([value, participant_embedding], dim=-1)
        update_not_chosen, _ = self.call_subnetwork('x_C_nc', inputs)
        
        # next_state += state_update_chosen * action + state_update_not_chosen * (1-action)
        next_value += update_chosen * action + update_not_chosen * (1-action)
        next_value = self._sigmoid(next_value)
        
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
        
        if batch_first:
            inputs = inputs.permute(1, 0, 2)
        
        action_array = inputs[:, :, :self._n_actions].float()
        reward_array = inputs[:, :, self._n_actions:-1].float()
        participant_id_array = inputs[:, :, -1].unsqueeze(-1).int()
           
        if prev_state is not None:
            self.set_state(prev_state[0], prev_state[1], prev_state[2], prev_state[3], prev_state[4], prev_state[5])
        else:
            self.set_initial_state(batch_size=inputs.shape[1])
        
        # get previous model state
        state = [s.squeeze(1) for s in self.get_state()]  # remove model dim for forward pass -> only one model
        reward_state, choice_state, uncertainty_state, reward_value, choice_value, uncertainty_value = state
        
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
            reward_value_prev, choice_value_prev, uncertainty_value_prev = reward_value.clone(), choice_value.clone(), uncertainty_value.clone() 
            reward_value, learning_rate, reward_state = self.value_network(reward_state, reward_value, action, reward, participant)
            choice_value, choice_state = self.choice_network(choice_state, choice_value, action, repeated, participant)
            
            logits_array[timestep, :, :] += reward_value * beta_reward + choice_value * beta_choice
            
            self._prev_action = action.clone()
            
            # signal tracking to collect training samples for SINDy
            if not torch.is_grad_enabled():
                # control signals
                self.append_timestep_sample('c_a', action)  # action
                self.append_timestep_sample('c_r', reward[:, :1] if not self._counterfactual else reward[:, action.argmax(dim=-1)])  # reward chosen
                self.append_timestep_sample('c_r_cf', reward[:, 1:] if not self._counterfactual else reward[:, (1-action).argmax(dim=-1)])  # reward not-chosen (aka counterfactual)
                self.append_timestep_sample('c_a_repeat', repeated)  # action repeated
                self.append_timestep_sample('c_V', reward_value_prev)  # reward-based values
                
                # dynamical variabels
                self.append_timestep_sample('x_V_LR', torch.zeros_like(learning_rate), learning_rate)
                self.append_timestep_sample('x_V_LR_cf', torch.zeros_like(learning_rate), learning_rate if reward[:, -1].mean() != -1 else torch.zeros_like(learning_rate))
                self.append_timestep_sample('x_V_nc', reward_value_prev, reward_value)
                self.append_timestep_sample('x_C', choice_value_prev, choice_value)
                self.append_timestep_sample('x_C_nc', choice_value_prev, choice_value)

        # add model dim again and set state
        self.set_state(reward_state.unsqueeze(1), choice_state.unsqueeze(1), uncertainty_state.unsqueeze(1), reward_value.unsqueeze(1), choice_value.unsqueeze(1), uncertainty_value.unsqueeze(1))
        
        if batch_first:
            logits_array = logits_array.permute(1, 0, 2)
            
        return logits_array, self.get_state()

    def set_initial_state(self, batch_size: int=1, return_dict=False):
        self._prev_action = torch.zeros(batch_size, self._n_actions, device=self.device)
        return super().set_initial_state(batch_size, return_dict)
