import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple


class baseRNN(nn.Module):
    def __init__(self, n_actions, hidden_size, init_value=0.5):
        super(baseRNN, self).__init__()
        
        self.init_value = init_value
        self._n_actions = n_actions
        self._hidden_size = hidden_size
               
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, *args):
        raise NotImplementedError('This method is not implemented.')
    
    def initial_state(self, batch_size=1, device=None):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.set_state(
            torch.zeros([batch_size, self._hidden_size], dtype=torch.float).to(device),
            torch.zeros([batch_size, self._hidden_size], dtype=torch.float).to(device),
            torch.zeros([batch_size, self._n_actions], dtype=torch.float).to(device),
            (self.init_value + torch.zeros([batch_size, self._n_actions], dtype=torch.float)).to(device)
            )
        
        return self.get_state()
        
    def set_state(self, habit_state: torch.Tensor, value_state: torch.Tensor, habit: torch.Tensor, value: torch.Tensor):
        """this method sets the hidden state
        
        Args:
            state (Tuple[torch.Tensor]): hidden state
        """
        
        self._state = tuple([habit_state, value_state, habit, value])
        
    def get_state(self):
        """this method returns the hidden state
        
        Returns:
            Tuple[torch.Tensor]: tuple of habit state, value state, habit, value
        """
        
        return self._state
    
    
class HybRNN(baseRNN):
    def __init__(
        self,
        n_actions, 
        hidden_size,
        init_value=0.5,
        use_habit=False,
        last_output=False,
        last_state=False,
        ):
        
        super(HybRNN, self).__init__(n_actions, hidden_size, init_value)
        
        # define level of recurrence
        self._vs, self._hs, self._vo, self._ho, self._wh = last_state, last_state, last_output, last_output, use_habit
        
        # define general network parameters
        self.init_value = init_value
        self._n_actions = n_actions
        # self._w_h = 0 if not use_habit else 1#nn.Parameter(torch.randn(1))
        # self._w_v = 1
        # self.beta = 3#nn.Parameter(torch.tensor(1., dtype=torch.float32).reshape((1, 1)))
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
        
        # value network
        self.reward_blind_update = nn.Linear(n_actions, n_actions)
        self.hidden_layer_value = nn.Linear(input_size, hidden_size)
        self.reward_based_update = nn.Linear(hidden_size, 1)
        
        # habit network
        self.hidden_layer_habit = nn.Linear(input_size, hidden_size)
        self.habit_layer = nn.Linear(hidden_size, n_actions)
        
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

        # first reward-blind mechanism (forgetting) for all elements
        blind_update = self.reward_blind_update(value)
        
        # now reward-based update for the chosen element        
        # get the value of the chosen action
        chosen_value = torch.sum(blind_update * action, dim=-1).view(-1, 1)
        inputs = torch.cat([chosen_value, reward], dim=-1).float()
        
        if self._vo:
            inputs = torch.cat([inputs, blind_update], dim=-1).float()
            
        if self._vs:
            inputs = torch.cat([inputs, state], dim=-1)
        
        next_state = self.tanh(self.hidden_layer_value(inputs))
        reward_update = self.reward_based_update(next_state)
        
        next_value = action * reward_update + (1-action) * blind_update

        return next_value, next_state
    
    def habit_network(self, state, habit, action):
        """this method computes the action-based updates for the Q-Values without considering the reward
        
        Args:
            state (torch.Tensor): last hidden state
            habit (torch.Tensor): last habit
            action (torch.Tensor): chosen action

        Returns:
            torch.Tensor: updated habit
        """
        
        inputs = action
        if self._ho:
            inputs = torch.cat([inputs, habit], dim=-1)
        if self._hs:
            inputs = torch.cat([inputs, state], dim=-1)
        
        next_state = self.tanh(self.hidden_layer_habit(inputs))
        next_habit = self.habit_layer(next_state)
        
        return next_habit, next_state
    
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
        logits = torch.zeros(inputs.shape[0], inputs.shape[1], self._n_actions, device=inputs.device)
        
        # check if action is one-hot encoded
        if action.shape[-1] == 1:
            action = F.one_hot(action.squeeze(1).long(), num_classes=self._n_actions).float()
            
        if prev_state is not None:
            self.set_state(*prev_state)
        else:
            self.initial_state(batch_size=inputs.shape[1], device=inputs.device)
        h_state, v_state, habit, value = self.get_state()
        
        for t, a, r in zip(timesteps, action, reward):            
            # compute the updates
            value, v_state = self.value_network(v_state, value, a, r)
            logit = value
            if self._wh:
                habit, h_state = self.habit_network(h_state, habit, a)
                logit += habit
                
            logits[t, :, :] = logit.clone()
        
        # set state
        self.set_state(h_state, v_state, habit, value)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
            
        return logits, self.get_state()
    
    
class LSTM(baseRNN):
    def __init__(self, n_actions, hidden_size, init_value=0.5):
        super(LSTM, self).__init__(n_actions, hidden_size, init_value)
        
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
            self.initial_state(batch_size=inputs.shape[1], device=inputs.device)
        c0, h0, _, _ = self.get_state()
        
        # forward pass
        lstm_out, (c, h) = self.lstm(torch.concat((action, reward), dim=-1), (c0.unsqueeze(0), h0.unsqueeze(0)))
        logits = self.output_layer(lstm_out)
        
        self.set_state(c.squeeze(0), h.squeeze(0), torch.zeros([inputs.shape[1], self._n_actions], dtype=torch.float).to(inputs.device), logits)
        
        if batch_first:
            logits = logits.permute(1, 0, 2)
        
        return logits, self.get_state()