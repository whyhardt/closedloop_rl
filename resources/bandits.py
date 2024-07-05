"""Environments + agents for 2-armed bandit task."""
# pylint: disable=line-too-long
from typing import NamedTuple, Union, Optional, List, Dict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy, deepcopy
import torch
from torch import nn
import torch.utils

from resources.rnn import RLRNN, EnsembleRNN
from rnn_utils import DatasetRNN

# Setup so that plots will look nice
small = 15
medium = 18
large = 20
plt.rc('axes', titlesize=large)
plt.rc('axes', labelsize=medium)
plt.rc('xtick', labelsize=small)
plt.rc('ytick', labelsize=small)
plt.rc('legend', fontsize=small)
plt.rc('figure', titlesize=large)
mpl.rcParams['grid.color'] = 'none'
mpl.rcParams['axes.facecolor'] = 'white'
plt.rcParams['svg.fonttype'] = 'none'

###################################
# CONVENIENCE FUNCTIONS.          #
###################################


def _check_in_0_1_range(x, name):
  if not (0 <= x <= 1):
    raise ValueError(
        f'Value of {name} must be in [0, 1] range. Found value of {x}.')


###################################
# GENERATIVE FUNCTIONS FOR AGENTS #
###################################


class AgentQ:
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

  def __init__(
      self,
      alpha: float = 0.2,
      beta: float = 3.,
      n_actions: int = 2,
      forget_rate: float = 0.,
      perseverance_bias: float = 0.,
      correlated_reward: bool = False,
      ):
    """Update the agent after one step of the task.

    Args:
      alpha: scalar learning rate
      beta: scalar softmax inverse temperature parameter.
      n_actions: number of actions (default=2)
      forgetting_rate: rate at which q values decay toward the initial values (default=0)
      perseveration_bias: rate at which q values move toward previous action (default=0)
    """
    self._prev_choice = -1
    self._alpha = alpha
    self._beta = beta
    self._n_actions = n_actions
    self._forget_rate = forget_rate
    self._perseverance_bias = perseverance_bias
    self._correlated_reward = correlated_reward
    self._q_init = 0.5
    self.new_sess()

    _check_in_0_1_range(alpha, 'alpha')
    _check_in_0_1_range(forget_rate, 'forget_rate')

  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init * np.ones(self._n_actions)
    self._prev_choice = -1

  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    decision_variable = np.exp(self.q * self._beta)
    choice_probs = decision_variable / np.sum(decision_variable)
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs()
    choice = np.random.choice(self._n_actions, p=choice_probs)
    return choice

  def update(self,
             choice: int,
             reward: float):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    
    # Forgetting - restore q-values of non-chosen actions towards the initial value
    non_chosen_action = np.arange(self._n_actions) != choice
    self._q[non_chosen_action] = (1-self._forget_rate) * self._q[non_chosen_action] + self._forget_rate * self._q_init

    # Reward-based update - Update chosen q for chosen action with observed reward
    q_reward_update = (1 - self._alpha) * self._q[choice] + self._alpha * reward
    
    # Correlated update - Update non-chosen q for non-chosen action with observed reward
    if self._correlated_reward:
      # index_correlated_update = self._n_actions - choice - 1
      # self._q[index_correlated_update] = (1 - self._alpha) * self._q[index_correlated_update] + self._alpha * (1 - reward) 
      # alternative implementation - not dependent on reward but on reward-based update
      index_correlated_update = self._n_actions - choice - 1
      self._q[index_correlated_update] = (1 - self._alpha) * self._q[index_correlated_update] - self._alpha * (self._q[choice] - q_reward_update) 
    
    # Memorize current choice for perseveration
    self._prev_choice = choice
    
    self._q[choice] = q_reward_update
    
  @property
  def q(self):
    q = self._q.copy()
    if self._prev_choice != -1:
      q[self._prev_choice] += self._perseverance_bias
    return q


class AgentQuadQ(AgentQ):
  
  def __init__(
      self,
      alpha: float=0.2,
      beta: float=3.,
      n_actions: int=2,
      forgetting_rate: float=0.,
      perseveration_bias: float=0.,
      ):
    super().__init__(alpha, beta, n_actions, forgetting_rate, perseveration_bias)
  
  def update(self,
            choice: int,
            reward: float):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    
    # Decay q-values toward the initial value.
    self._q = (1-self._forget_rate) * self._q + self._forget_rate * self._q_init

    # Update chosen q for chosen action with observed reward.
    self._q[choice] = self._q[choice] - self._alpha * self._q[choice]**2 + self._alpha * reward


class AgentSindy:

  def __init__(
      self,
      n_actions: int=2,
      beta: float=1.,
      ):

    self._q_init = 0.5
    self._beta = beta
    self._n_actions = n_actions
    self._prev_choice = None
        
    self._update_rule = lambda q, choice, reward: q[choice] + reward
    self._update_rule_formula = None
    
  def set_update_rule(self, update_rule: callable, update_rule_formula: str=None):
    self._update_rule=update_rule
    self._update_rule_formula=update_rule_formula

  @property
  def update_rule(self):
    if self._update_rule_formula is not None:
      return self._update_rule_formula
    else:
      return f'{self._update_rule}'

  def update(self, choice: int, reward: int):
    for c in range(self._n_actions):
      q, h = self._update_rule(self._q[c], self._h[c], int(choice==c), int(self._prev_choice==c), reward)
      self._q[c] = q
      self._h[c] = h
    self._prev_choice = choice
    
  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init + np.zeros(self._n_actions)
    self._h = np.zeros(self._n_actions)
    self._prev_choice = -1
    
  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    decision_variable = np.exp(self.q * self._beta)
    choice_probs = decision_variable / np.sum(decision_variable)
    return choice_probs

  def get_choice(self, random=True) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs()
    if random:
      choice = np.random.choice(self._n_actions, p=choice_probs)
    else:
      choice = np.argmax(choice_probs)
    return choice
  
  @property
  def q(self):
    return (self._q + self._h).copy()


class AgentNetwork:
    """A class that allows running a pretrained RNN as an agent.

    Attributes:
        model: A PyTorch module representing the RNN architecture
    """

    def __init__(
      self,
      model: Union[RLRNN, EnsembleRNN],
      n_actions: int = 2,
      device = torch.device('cpu'),
      ):
        """Initialize the agent network.

        Args:
            model: A PyTorch module representing the RNN architecture
            n_actions: number of permitted actions (default = 2)
        """
        
        self._q_init = 0.5
        if device != model.device:
          model = model.to(device)
        if isinstance(model, RLRNN):
          self._model = RLRNN(model._n_actions, model._hidden_size, model.init_value, model._vo, model._vs, list(model.history.keys()), device=model.device).to(model.device)
          self._model.load_state_dict(model.state_dict())
        else:
          self._model = model
        self._model.eval()
        self._xs = torch.zeros((1, 2))-1
        self._n_actions = n_actions
        self.new_sess()

    def new_sess(self):
      """Reset the network for the beginning of a new session."""
      # self.set_state(self._model.initial_state(batch_size=1, return_dict=True))
      self._model.initial_state(batch_size=1, return_dict=True)
      self._xs = torch.zeros((1, 2))-1
    
    # def set_state(self, state: Dict[str, torch.Tensor]):
    #   state = [v for k, v in state.items() if not 'hidden' in k]  # get only the non-hidden states i.e. habit and value
    #   self._state = tuple([tensor.cpu().numpy() for tensor in state])
    
    def get_value(self):
      """Return the value of the agent's current state."""
      state = self._model.get_state()[-1].cpu().numpy()
      value = state[:, 0].reshape(-1)
      return value
    
    def get_choice_probs(self) -> np.ndarray:
      """Predict the choice probabilities as a softmax over output logits."""
      # choice_probs = torch.nn.functional.softmax(self.get_value(), dim=-1).view(-1)
      value = self.get_value()
      # if all(np.abs(value) > 10):
      #   choice_probs = np.array([0.5, 0.5])
      # elif any(value > 10):
      #   # TODO: this works currently only for 2 actions
      #   choice_probs = np.zeros(self._n_actions)
      #   choice_probs[np.argmax(value)] = 1
      # else:
      choice_probs = np.exp(self.get_value()) / np.sum(np.exp(self.get_value()))
      return choice_probs

    def get_choice(self, random=True):
      """Sample choice."""
      choice_probs = self.get_choice_probs()
      if random:
        choice = np.random.choice(self._n_actions, p=choice_probs)
      else:
        choice = np.argmax(choice_probs)
      return choice

    def update(self, choice: float, reward: float):
      self._xs = torch.tensor([[choice, reward]], device=self._model.device)
      with torch.no_grad():
        self._model(self._xs, self._model.get_state())
        # self.set_state(self._model.get_state(return_dict=True))
            
    @property
    def q(self):
      return copy(self.get_value())


################
# ENVIRONMENTS #
################


class EnvironmentBanditsFlips:
  """Env for 2-armed bandit task with reward probs that flip in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.8,
      reward_prob_low: float = 0.2,
  ):
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    # Choose a random block to start in
    self._block = np.random.binomial(1, 0.5)
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Flip the reward probabilities for a new block."""
    # Flip the block
    self._block = 1 - self._block
    # Set the reward probabilites
    if self._block == 1:
      self.reward_probs = [self._reward_prob_high, self._reward_prob_low]
    else:
      self.reward_probs = [self._reward_prob_low, self._reward_prob_high]

  def step(self, choice):
    """Step the model forward given chosen action."""
    # Choose the reward probability associated with agent's choice
    reward_prob_trial = self.reward_probs[choice]

    # Sample a reward with this probability
    reward = float(np.random.binomial(1, reward_prob_trial))

    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    return float(reward)

  @property
  def n_actions(self) -> int:
    return 2


class EnvironmentBanditsDrift:
  """Environment for a drifting two-armed bandit task.

  Reward probabilities on each arm are sampled randomly between 0 and 1. On each
  trial, gaussian random noise is added to each.

  Attributes:
    sigma: A float, between 0 and 1, giving the magnitude of the drift
    reward_probs: Probability of reward associated with each action
    n_actions: number of actions available
  """

  def __init__(
      self,
      sigma: float,
      n_actions: int = 2,
      non_binary_reward: bool = False,
      correlated_reward: bool = False,
      ):
    """Initialize the environment."""
    # Check inputs
    if sigma < 0:
      msg = f'Argument sigma but must be greater than 0. Found: {sigma}.'
      raise ValueError(msg)

    # Initialize persistent properties
    self._sigma = sigma
    self._n_actions = n_actions
    self._non_binary_reward = non_binary_reward
    self._correlated_rewards = correlated_reward

    # Sample new reward probabilities
    self._new_sess()

  def _new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = np.random.rand(self._n_actions)

  def step(self, choice: int) -> int:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """
    # Check inputs
    if choice not in range(self._n_actions):
      msg = (
          f'Found value for choice of {choice}, but must be in '
          f'{list(range(self._n_actions))}')
      raise ValueError(msg)

    # Sample reward with the probability of the chosen side
    # reward = np.random.rand() < self._reward_probs[choice]
    if not self._non_binary_reward:
      reward = float(np.random.rand() < self._reward_probs[choice])
    else: 
      reward = self._reward_probs[choice]
    
    # Add gaussian noise to reward probabilities
    drift = np.random.normal(loc=0, scale=self._sigma, size=self._n_actions)
    if not self._correlated_rewards:
      self._reward_probs += drift
    else:
      for i in range(self._n_actions//2):
        self._reward_probs[i] += drift[i]
        self._reward_probs[i-1] -= drift[i]
        
    # Fix reward probs that've drifted below 0 or above 1
    self._reward_probs = np.clip(self._reward_probs, 0, 1)

    return reward

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return self._n_actions


class BanditSession(NamedTuple):
  """Holds data for a single session of a bandit task."""
  choices: np.ndarray
  rewards: np.ndarray
  timeseries: np.ndarray
  q: np.ndarray
  n_trials: int
  

Agent = Union[AgentQ, AgentNetwork]
Environment = Union[EnvironmentBanditsFlips, EnvironmentBanditsDrift]


###############
#  EXECUTION  #
###############


def run_experiment(
  agent: Agent,
  environment: Environment,
  n_trials: int,
  ) -> BanditSession:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_trials: The number of steps in the session you'd like to generate

  Returns:
    experiment: A BanditSession holding choices and rewards from the session
  """
  
  choices = np.zeros(n_trials+1)
  rewards = np.zeros(n_trials+1)
  qs = np.zeros((n_trials+1, environment.n_actions))
  reward_probs = np.zeros((n_trials+1, environment.n_actions))

  for trial in range(n_trials+1):
    # First record environment reward probs
    reward_probs[trial] = environment.reward_probs
    qs[trial] = agent.q
    # First agent makes a choice
    choice = agent.get_choice()
    # Then environment computes a reward
    reward = environment.step(choice)
    # Log choice and reward
    choices[trial] = choice
    rewards[trial] = reward
    # Agent learns
    agent.update(choice, reward)
    
  experiment = BanditSession(n_trials=n_trials,
                             choices=choices[:-1].astype(int),
                             rewards=rewards[:-1],
                             timeseries=reward_probs[:-1],
                             q=qs[:-1])
  return experiment, choices.astype(int), rewards


def create_dataset(
  agent: Agent,
  environment: Environment,
  n_trials_per_session: int,
  n_sessions: int,
  sequence_length: int = None,
  stride: int = 1,
  device=torch.device('cpu'),
  ):
  """Generates a behavioral dataset from a given agent and environment.

  Args:
    agent: An agent object to generate choices
    environment: An environment object to generate rewards
    n_trials_per_session: The number of trials in each behavioral session to
      be generated
    n_sessions: The number of sessions to generate
    batch_size: The size of the batches to serve from the dataset. If None, 
      batch_size defaults to n_sessions

  Returns:
    A torch.utils.data.Dataset object suitable for training the RNN object.
    An experliment_list with the results of (simulated) experiments
  """
  xs = np.zeros((n_trials_per_session, n_sessions, agent._n_actions + 1))
  ys = np.zeros((n_trials_per_session, n_sessions, agent._n_actions))
  experiment_list = []

  for session in range(n_sessions):
    agent.new_sess()
    experiment, choices, rewards = run_experiment(agent, environment, n_trials_per_session)
    experiment_list.append(experiment)
    # one-hot encoding of choices
    choices = np.eye(agent._n_actions)[choices]
    xs[:, session] = np.concatenate((choices[:-1], rewards[:-1].reshape(-1, 1)), axis=-1)
    ys[:, session] = choices[1:]

  dataset = DatasetRNN(
    xs=np.swapaxes(xs, 0, 1), 
    ys=np.swapaxes(ys, 0, 1),
    sequence_length=sequence_length,
    stride=stride,
    device=device)
  return dataset, experiment_list


def get_update_dynamics(experiment: BanditSession, agent: Union[AgentQ, AgentNetwork, AgentSindy]):
  """Compute Q-Values of a specific agent for a specific experiment sequence with given actions and rewards.

  Args:
      experiment (BanditSession): _description_
      agent (_type_): _description_

  Returns:
      _type_: _description_
  """
  
  choices = np.expand_dims(experiment.choices, 1)
  rewards = np.expand_dims(experiment.rewards, 1)
  qs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  choice_probs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  
  agent.new_sess()
  
  for trial in range(experiment.choices.shape[0]):
    qs[trial] = agent.q
    choice_probs[trial] = agent.get_choice_probs()
    agent.update(int(choices[trial]), float(rewards[trial]))
    
  return qs, choice_probs


###############
# DIAGNOSTICS #
###############


def plot_session(
  choices: np.ndarray,
  rewards: np.ndarray,
  timeseries: List[np.ndarray],
  timeseries_name: str,
  labels: Optional[List[str]] = None,
  title: str = '',
  x_label = 'Trials',
  fig_ax = None,
  compare=False,
  color=None,
  binary=False,
  axis_info=True,
  ):
  """Plot data from a single behavioral session of the bandit task.

  Args:
    choices: The choices made by the agent
    rewards: The rewards received by the agent
    timeseries: The reward probabilities on each arm
    timeseries_name: The name of the reward probabilities
    labels: The labels for the lines in the plot
    fig_ax: A tuple of a matplotlib figure and axis to plot on
    compare: If True, plot multiple timeseries on the same plot
    color: A list of colors to use for the plot; at least as long as the number of timeseries
  """

  if color == None:
    color = [None]*len(timeseries)
  
  choose_high = choices == 1
  choose_low = choices == 0
  rewarded = rewards == 1
  y_high = np.max(timeseries) + 0.1
  y_low = np.min(timeseries) - 0.1
  
  # Make the plot
  if fig_ax is None:
    fig, ax = plt.subplots(1, figsize=(10, 3))
  else:
    fig, ax = fig_ax
    
  if compare:
    if timeseries.ndim==2:
      timeseries = np.expand_dims(timeseries, -1)
    if timeseries.ndim!=3 or timeseries.shape[-1]!=1:
      raise ValueError('If compare: timeseries must be of shape (agent, timesteps, 1).')
  else:
    if timeseries.ndim!=2:
      raise ValueError('timeseries must be of shape (timesteps, n_actions).')
                       
  if not compare:
    choices = np.expand_dims(choices, 0)
    timeseries = np.expand_dims(timeseries, 0)
  
  for i in range(timeseries.shape[0]):
    if labels is not None:
      if timeseries[i].ndim == 1:
        timeseries[i] = timeseries[i, :, None]
      if not compare:
        if len(labels) != timeseries[i].shape[1]:
          raise ValueError('labels length must match timeseries.shape[1].')
      else:
        if timeseries[i].shape[1] != 1:
          raise ValueError('If compare: timeseries.shape[1] must be 1.')
        if len(labels) != timeseries.shape[0]:
          raise ValueError('If compare: labels length must match timeseries.shape[0].')
      for ii in range(timeseries[i].shape[-1]):
          label = labels[ii] if not compare else labels[i]
          ax.plot(timeseries[i, :, ii], label=label, color=color[i])
      ax.legend(bbox_to_anchor=(1, 1))
    else:  # Skip legend.
      ax.plot(timeseries[i], color=color[i])

  if choices.max() <= 1 and binary:
    # Rewarded high
    ax.scatter(
        np.argwhere(choose_high & rewarded),
        y_high * np.ones(np.sum(choose_high & rewarded)),
        color='green',
        marker=3)
    ax.scatter(
        np.argwhere(choose_high & rewarded),
        y_high * np.ones(np.sum(choose_high & rewarded)),
        color='green',
        marker='|')
    # Omission high
    ax.scatter(
        np.argwhere(choose_high & 1 - rewarded),
        y_high * np.ones(np.sum(choose_high & 1 - rewarded)),
        color='red',
        marker='|')

    # Rewarded low
    ax.scatter(
        np.argwhere(choose_low & rewarded),
        y_low * np.ones(np.sum(choose_low & rewarded)),
        color='green',
        marker='|')
    ax.scatter(
        np.argwhere(choose_low & rewarded),
        y_low * np.ones(np.sum(choose_low & rewarded)),
        color='green',
        marker=2)
    # Omission Low
    ax.scatter(
        np.argwhere(choose_low & 1 - rewarded),
        y_low * np.ones(np.sum(choose_low & 1 - rewarded)),
        color='red',
        marker='|')

  if axis_info:
    ax.set_xlabel(x_label)
    ax.set_ylabel(timeseries_name)
    ax.set_title(title)
  else:
    ax.set_xticks(np.linspace(1, len(choices), 6))
    ax.set_xticklabels(['']*6)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(['']*5)
    

def show_valuemetric(experiment_list, label=None):
  """Plot value metric over time from data in experiment_list."""
  if experiment_list is None:
      print(('Skipping experiment because None value was found for experiment_lists.\n'
             'This is usually the case when using imported real data.'))
      return
  
  reward_prob_bins = np.linspace(-1, 1, 50)
  n_left = np.zeros(len(reward_prob_bins)-1)
  n_right = np.zeros(len(reward_prob_bins)-1)

  for sessdata in experiment_list:
    reward_prob_diff = sessdata.timeseries[:, 0] - sessdata.timeseries[:, 1]
    for reward_prob_i in range(len(n_left)):
      trials_in_bin = np.logical_and(
          (reward_prob_bins[reward_prob_i] < reward_prob_diff) ,
          (reward_prob_diff < reward_prob_bins[reward_prob_i+1]))
      n_left[reward_prob_i] += np.sum(
          np.logical_and(trials_in_bin, sessdata.choices == 0.))
      n_right[reward_prob_i] += np.sum(
          np.logical_and(trials_in_bin, sessdata.choices == 1.))

  choice_probs = n_left / (n_left + n_right)

  xs = reward_prob_bins[:-1] - (reward_prob_bins[1]-reward_prob_bins[0])/2
  ys = choice_probs
  plt.plot(xs, ys, label=label)
  plt.ylim((0, 1))
  plt.xlabel('Difference in Reward Probability (left - right)')
  plt.ylabel('Proportion of Leftward Choices')


def show_total_reward_rate(experiment_list):
  if experiment_list is None:
      print('Skipping showing reward rate for Non experiment_list')
      return
  rewards = 0
  trials = 0

  for sessdata in experiment_list:
    rewards += np.sum(sessdata.rewards)
    trials += sessdata.n_trials

  reward_rate = 100*rewards/trials
  print(f'Total Reward Rate is: {reward_rate:0.3f}%')
