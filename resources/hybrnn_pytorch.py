import torch
import torch.nn as nn
import torch.nn.functional as F


class BiRNN(nn.Module):
    """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

    def __init__(self, rl_params, network_params, init_value=0.5):
        super(BiRNN, self).__init__()

        self.hs = rl_params['s']
        self.vs = rl_params['s']
        self.ho = rl_params['o']
        self.vo = rl_params['o']

        self.w_h = rl_params['w_h']
        self.w_v = rl_params['w_v']
        self.init_value = init_value

        self.n_actions = network_params['n_actions']
        self.hidden_size = network_params['hidden_size']

        if rl_params['fit_forget']:
            self.register_parameter('unsigmoid_forget', nn.Parameter(torch.randn(1)))
        else:
            self.forget = rl_params['forget']

        self.forget = torch.sigmoid(self.unsigmoid_forget) if rl_params['fit_forget'] else rl_params['forget']

        self.linear_value = nn.Linear(self.n_actions, self.hidden_size)
        self.linear_habit = nn.Linear(self.n_actions, self.hidden_size)
        
        self.reward_blind_update = nn.Linear(self.n_actions, self.n_actions)
        
        self.prev_state = self.initial_state()

    def _value_rnn(self, state, value, action, reward):
        blind_update = self.reward_blind_update(value)
        inputs = torch.cat([blind_update * action, reward.unsqueeze(-1)], dim=-1)

        if self.vo:
            inputs = torch.cat([inputs, value], dim=-1)
        if self.vs:
            inputs = torch.cat([inputs, state], dim=-1)

        next_state = F.linear(inputs, self.linear_value.weight, self.linear_value.bias)

        reward_update = F.linear(next_state, self.linear_value.weight, self.linear_value.bias)

        next_value = action * reward_update + (1 - action) * blind_update

        return next_value, next_state

    def _habit_rnn(self, state, habit, action):
        inputs = action

        if self.ho:
            inputs = torch.cat([inputs, habit], dim=-1)
        if self.hs:
            inputs = torch.cat([inputs, state], dim=-1)

        next_state = F.linear(inputs, self.linear_habit.weight, self.linear_habit.bias)
        next_habit = F.linear(next_state, self.linear_habit.weight, self.linear_habit.bias)

        return next_habit, next_state

    def forward(self, inputs):        
        h_state, v_state, habit, value = self.prev_state
        action = inputs[:, :, :-1]
        reward = inputs[:, :, -1]

        if action.shape[-1] != self.n_actions:
            action = F.one_hot(action[:, 0], self.n_actions).float()

        next_value, next_v_state = self._value_rnn(v_state, value, action, reward)
        next_habit, next_h_state = self._habit_rnn(h_state, habit, action)

        logits = self.w_v * next_value + self.w_h * next_habit
        
        self.prev_state = (next_h_state, next_v_state, next_habit, next_value)

        return logits, (next_h_state, next_v_state, next_habit, next_value)

    def initial_state(self, batch_size=None):
        if batch_size is None:
            # raise ValueError("Batch size must be provided.")
            batch_size = 32
            
        return (
            torch.zeros(batch_size, self.hidden_size),  # h_state
            torch.zeros(batch_size, self.hidden_size),  # v_state
            torch.zeros(batch_size, self.n_actions),  # habit
            self.init_value * torch.ones(batch_size, self.n_actions),  # value
        )
