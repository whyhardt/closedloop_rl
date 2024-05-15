import torch
import torch.nn as nn
import torch.functional as F

from typing import Optional, Tuple

class RNN(nn.Module):
    def __init__(
        self, 
        n_actions, 
        hidden_size,
        init_value=0.5,
        use_habit=False,
        last_output=False,
        last_state=False,
        ):
        
        super(RNN, self).__init__()
        
        # define level of recurrence
        self._vs, self._hs, self._vo, self._ho = last_state, last_state, last_output, last_output
        
        # define general network parameters
        self.init_value = 0.5
        self._n_actions = n_actions
        self._w_h = 0 if not use_habit else nn.Parameter(torch.randn(1))
        self._w_v = 1
        
        # define layer parameters
        self._hidden_size = hidden_size
        input_size = self._n_actions + 1
        if self._vo:
            input_size += self._n_actions
        if self._vs:
            input_size += self._hidden_size
        
        # define layers
        # activation functions
        self.activation = nn.Tanh()
        
        # value network
        self.reward_blind_update = nn.Linear(n_actions, n_actions)
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.reward_based_update = nn.Linear(hidden_size, 1)
        
        # habit network
        self.hidden_layer = nn.Linear(input_size, hidden_size)
        self.habit_layer = nn.Linear(hidden_size, n_actions)
        
    def value_network(self, state, value, action, reward):
        """this method computes the reward-blind and reward-based updates for the Q-Values without considering the habit (e.g. last chosen action)
        

        Args:
            state (torch.Tensor): last hidden state
            value (torch.Tensor): last Q-Values
            action (torch.Tensor): chosen action
            reward (torch.Tensor): received reward

        Returns:
            torch.Tensor: updated Q-Values
        """
        
        # first action-reward-blind mechanism (forgetting) for all elements
        blind_update = self.reward_blind_update(value)
        
        # now update of only the chosen element
        inputs = torch.cat([blind_update * action, reward], dim=-1)
        if self._vo:
            inputs = torch.cat([inputs, value], dim=-1)
        if self._vs:
            inputs = torch.cat([inputs, state], dim=-1)
        
        next_state = self.activation(self.hidden_layer(inputs))
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
        
        next_state = self.activation(self.hidden_layer(inputs))
        next_habit = self.habit_layer(next_state)
        
        return next_habit, next_state
    
    def forward(self, inputs: torch.Tensor, prev_state: Optional[Tuple[torch.Tensor]] = None):
        """this method computes the next hidden state and the updated Q-Values based on the input and the previous hidden state
        
        Args:
            inputs (torch.Tensor): input tensor
            prev_state (torch.Tensor): previous state of form (h_state, v_state, habit, value)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: updated Q-Values and hidden state
        """
        
        if prev_state is None:
            prev_state = self.get_state()
        
        h_state, v_state, habit, value = prev_state
        action = inputs[:, :, :-1]
        reward = inputs[:, :, -1]
        
        # check if action is one-hot encoded
        if action.shape[-1] != self._n_actions:
            action = F.one_hot(action[:, 0], self._n_actions).float()
        
        # compute the updates
        value, v_state = self.value_network(v_state, value, action, reward)
        if self._w_h > 0:
            habit, h_state = self.habit_network(h_state, habit, action)
        
        # combine value and habit
        logits = self._w_v * value + self._w_h * habit
        
        # set state
        self.set_state((h_state, v_state, habit, value))
        
        return logits, (h_state, v_state, habit, value)
    
    def initital_state(self, batch_size=1):
        """this method initializes the hidden state
        
        Args:
            batch_size (int, optional): batch size. Defaults to 1.

        Returns:
            Tuple[torch.Tensor]: initial hidden state
        """
        init_state = (
            torch.zeros([batch_size, self._hidden_size]),
            torch.zeros([batch_size, self._hidden_size]),
            torch.zeros([batch_size, self._n_actions]),
            self.init_value + torch.zeros([batch_size, self._n_actions]),
            )
        
        self.set_state(init_state)
        
        return self.get_state()
        
    def set_state(self, state):
        """this method sets the hidden state
        
        Args:
            state (Tuple[torch.Tensor]): hidden state
        """
        
        self._state = state
        
    def get_state(self):
        """this method returns the hidden state
        
        Returns:
            Tuple[torch.Tensor]: hidden state
        """
        
        return self._state