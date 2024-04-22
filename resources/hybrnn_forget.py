"""Define hybRNNs."""
from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp

RNNState = jnp.array


class BiRNN(hk.RNNCore):
  """A hybrid RNN: "habit" processes action choices; "value" processes rewards."""

  def __init__(self, rl_params, network_params, init_value=0.5):

    super().__init__()

    self._hs = rl_params['s']
    self._vs = rl_params['s']
    self._ho = rl_params['o']
    self._vo = rl_params['o']

    self.w_h = rl_params['w_h']
    self.w_v = rl_params['w_v']
    self.init_value = init_value

    self._n_actions = network_params['n_actions']
    self._hidden_size = network_params['hidden_size']

    if rl_params['fit_forget']:
      init = hk.initializers.RandomNormal(stddev=1, mean=0)
      self.forget = jax.nn.sigmoid(  # 0 < forget < 1
          hk.get_parameter('unsigmoid_forget', (1,), init=init)
      )
    else:
      self.forget = rl_params['forget']
    
  def _value_rnn(self, state, value, action, reward):

    # first action-reward-blind mechanism (forgetting) for all elements
    blind_update = hk.Linear(self._n_actions)(value)  #  * (1-action)

    # now update of only the chosen element
    inputs = jnp.concatenate(
        [blind_update * action, reward[:, jnp.newaxis]], axis=-1)#, action_i[:, jnp.newaxis]], axis=-1)

    if self._vo:  # "o" = output -> feed previous output back in
      inputs = jnp.concatenate([inputs, value], axis=-1)
    if self._vs:  # "s" = state -> feed previous hidden state back in
      inputs = jnp.concatenate([inputs, state], axis=-1)
    
    next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
    # next_state = hk.Linear(self._hidden_size)(inputs)

    # compute reward based update for chosen action
    reward_update = hk.Linear(1)(next_state)

    next_value = action * reward_update + (1-action) * blind_update
    
    return next_value, next_state

  def _habit_rnn(self, state, habit, action):

      inputs = action
      if self._ho:  # "o" = output -> feed previous output back in
        inputs = jnp.concatenate([inputs, habit], axis=-1)
      if self._hs:  # "s" = state -> feed previous hidden state back in
        inputs = jnp.concatenate([inputs, state], axis=-1)

      next_state = jax.nn.tanh(hk.Linear(self._hidden_size)(inputs))
      next_habit = hk.Linear(self._n_actions)(next_state)

      return next_habit, next_state

  def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
    h_state, v_state, habit, value = prev_state
    action = inputs[:, :-1]  # shape: (batch_size, )
    reward = inputs[:, -1]  # shape: (batch_size,)
    # check if action is one-hot encoded
    if action.shape[-1] != self._n_actions:
      action = jax.nn.one_hot(action[:, 0], self._n_actions)
    
    # Value module: update/create new values
    next_value, next_v_state = self._value_rnn(v_state, value, action, reward)

    # Habit module: update/create new habit
    next_habit, next_h_state = self._habit_rnn(h_state, habit, action)

    # Combine value and habit
    logits = self.w_v * next_value + self.w_h * next_habit  # (bs, n_a)

    return logits, (next_h_state, next_v_state, next_habit, next_value)

  def initial_state(self, batch_size: Optional[int]):

    return (
        0 * jnp.ones([batch_size, self._hidden_size]),  # h_state
        0 * jnp.ones([batch_size, self._hidden_size]),  # v_state
        0 * jnp.ones([batch_size, self._n_actions]),  # habit
        self.init_value * jnp.ones([batch_size, self._n_actions]),  # value
        )


class Lstm(hk.RNNCore):
  """LSTM that predicts action logits based on all inputs (action, reward)."""

  def __init__(self, n_hiddens, n_bandits):

    super().__init__()

    self._hidden_size = n_hiddens
    self._n_actions = n_bandits

  def __call__(self, inputs: jnp.array, prev_state):

    hidden_state, cell_state = prev_state

    forget_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    input_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    candidates = jax.nn.tanh(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )

    next_cell_state = forget_gate * cell_state + input_gate * candidates

    output_gate = jax.nn.sigmoid(
        hk.Linear(self._hidden_size, with_bias=False)(inputs)
        + hk.Linear(self._hidden_size)(hidden_state)
    )
    next_hidden_state = output_gate * jax.nn.tanh(next_cell_state)

    action_probs = jax.nn.softmax(
        hk.Linear(self._n_actions)(next_hidden_state))  # (batch_size, n_act)

    return action_probs, (next_hidden_state, next_cell_state)

  def initial_state(self, batch_size):

    return (jnp.zeros((batch_size, self._hidden_size)),  # hidden_state
            jnp.zeros((batch_size, self._hidden_size)))  # cell_state