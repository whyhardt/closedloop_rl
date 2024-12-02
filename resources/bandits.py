"""Environments + agents for 2-armed bandit task."""
# pylint: disable=line-too-long
from typing import NamedTuple, Union, Optional, Dict, Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy, deepcopy
import torch
from torch import nn
import torch.utils
import pysindy as ps

from resources.rnn import RLRNN, EnsembleRNN
from resources.rnn_utils import DatasetRNN

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
      n_actions: int = 2,
      alpha: float = 0.2,
      beta: float = 3.,
      forget_rate: float = 0.,
      perseverance_bias: float = 0.,
      alpha_perseverance: float = 1.,
      alpha_penalty: float = -1,
      confirmation_bias: float = 0.,
      parameter_variance: Union[Dict[str, float], float] = 0.,
      ):
    """Update the agent after one step of the task.
    
    Args:
      alpha (float): Baseline learning rate between 0 and 1.
      beta (float): softmax inverse noise temperature. Regulates the noise in the decision-selection.
      n_actions: number of actions (default=2)
      forget_rate (float): rate at which q values decay toward the initial values (default=0)
      perseverance_bias (float): rate at which q values move toward previous action (default=0)
      alpha_penalty (float): separate learning rate for negative outcomes
      confirmation_bias (float): higher learning rate for believe-confirming outcomes and lower learning rate otherwise
      parameter_variance (float): sets a variance around the model parameters' mean values to sample from a normal distribution e.g. at each new session. 0: no variance, -1: std = mean
    """
    
    self._mean_beta = beta
    self._mean_alpha = alpha
    self._mean_alpha_penalty = alpha_penalty if alpha_penalty >= 0 else alpha
    self._mean_confirmation_bias = confirmation_bias
    self._mean_forget_rate = forget_rate
    self._mean_perseverance_bias = perseverance_bias
    self._mean_alpha_perseverance = alpha_perseverance
    
    self._list_params = ['beta', 'alpha', 'alpha_penalty', 'confirmation_bias', 'forget_rate', 'perseverance_bias']
    self._beta = beta
    self._alpha = alpha
    self._alpha_penalty = alpha_penalty if alpha_penalty >= 0 else alpha
    self._confirmation_bias = confirmation_bias
    self._forget_rate = forget_rate
    self._perseverance_bias = perseverance_bias
    self._alpha_perseverance = alpha_perseverance
    
    self._n_actions = n_actions
    self._parameter_variance = self.check_parameter_variance(parameter_variance)
    self._q_init = 0.5
    
    self.new_sess()

    _check_in_0_1_range(alpha, 'alpha')
    _check_in_0_1_range(forget_rate, 'forget_rate')
    
    self._reward_prediction_error = lambda q, reward: reward-q
    
    self.new_sess()

  def check_parameter_variance(self, parameter_variance):
    if isinstance(parameter_variance, float):
      par_var_dict = {}
      for key in self._list_params:
        par_var_dict[key] = float
      parameter_variance = par_var_dict
    elif isinstance(parameter_variance, dict):
      # check that all keys in parameter_variance are valid
      not_valid_keys = []
      for key in parameter_variance:
        if not key in self._list_params:
          not_valid_keys.append(key)
      if len(not_valid_keys) > 0:
        raise ValueError(f'Some keys in parameter_variance are not valid ({not_valid_keys}). Valid keys are {self._list_params}')
      # check that all parameters are available - set to 0 if a parameter is not available
      for key in self._list_params:
        if not key in parameter_variance:
          parameter_variance[key] = 0.
    return parameter_variance
  
  def new_sess(self, sample_parameters=False):
    """Reset the agent for the beginning of a new session."""
    self._q = np.full(self._n_actions, self._q_init)
    self._c = np.zeros(self._n_actions)
    
    # sample new parameters
    if sample_parameters:
      sanity = False
      while not sanity:
        # sample new parameters until all sanity checks are passed
        self._beta = np.clip(np.random.normal(self._mean_beta, self._mean_beta/2 if self._parameter_variance['beta'] == -1 else self._parameter_variance['beta']), 0, 2*self._mean_beta)
        self._alpha = np.clip(np.random.normal(self._mean_alpha, self._mean_alpha/2 if self._parameter_variance['alpha'] == -1 else self._parameter_variance['alpha']), 0 , 1)
        self._alpha_penalty = np.clip(np.random.normal(self._mean_alpha_penalty, self._mean_alpha_penalty/2 if self._parameter_variance['alpha_penalty'] == -1 else self._parameter_variance['alpha_penalty']), 0, 1)
        self._confirmation_bias = np.clip(np.random.normal(self._mean_confirmation_bias, self._mean_confirmation_bias/2 if self._parameter_variance['confirmation_bias'] == -1 else self._parameter_variance['confirmation_bias']), 0, 1)
        self._forget_rate = np.clip(np.random.normal(self._mean_forget_rate, self._mean_forget_rate/2 if self._parameter_variance['forget_rate'] == -1 else self._parameter_variance['forget_rate']), 0, 1)
        self._perseverance_bias = np.clip(np.random.normal(self._mean_perseverance_bias, self._mean_perseverance_bias/2 if self._parameter_variance['perseverance_bias'] == -1 else self._parameter_variance['perseverance_bias']), 0, 1)
        self._alpha_perseverance = np.clip(np.random.normal(self._mean_alpha_perseverance, self._mean_alpha_perseverance/2 if self._parameter_variance['alpha_perseverance'] == -1 else self._parameter_variance['alpha_perseverance']), 0, 1)

        # sanity checks
        # 1. (alpha, alpha_penalty) + confirmation_bias*max_confirmation must be in range(0, 1)
        #     with max_confirmation = (q-q0)(r-q0) = +/- 0.25
        max_learning_rate = self._alpha + self._confirmation_bias*0.25 <= 1 and self._alpha_penalty + self._confirmation_bias*0.25 <= 1
        min_learning_rate = self._alpha + self._confirmation_bias*-0.25 >= 0 and self._alpha_penalty + self._confirmation_bias*-0.25 >= 0 
        sanity = max_learning_rate and min_learning_rate
      
  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    decision_variable = np.exp(self.q)
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
    
    # Reward-based updates
    # asymmetric learning rates
    alpha = self._alpha if reward > 0.5 else self._alpha_penalty
    
    # Reward-prediction-error
    rpe = self._reward_prediction_error(self._q[choice], reward)
    
    # Forgetting - restore q-values of non-chosen actions towards the initial value
    non_chosen_action = np.arange(self._n_actions) != choice
    forget_update = self._forget_rate * (self._q_init - self._q[non_chosen_action])
    
    # add confirmation bias to learning rate
    # Rollwage et al (2020): https://www.nature.com/articles/s41467-020-16278-6.pdf
    # if self._confirmation_bias:  
    # when any input to a cognitive mechanism is differentiable --> cognitive mechanism must be differentiable as well!
    # differentiable confirmation bias:
    alpha += self._confirmation_bias * (self._q[choice]-self._q_init)*(reward - 0.5)
      
      # full learning rate equation w/ confirmation bias = 0.5: 
      # (xLR)[k+1]    = 0.25 + 0.5 * (Q-0.5) * (Reward - 0.5)
      #               = 0.25 + 0.5*Q*Reward - 0.25*Q - 0.25*Reward + 0.125
      # SINDy target: = 0.375 - 0.25*Q - 0.25*Reward + 0.5*Q*Reward

    reward_update = alpha * rpe
    # Value_new = Value_old - lr * Value_old + lr * Reward 
    # Standard (static) with lr=0.25 --> Value_new = (1-0.25) * Value_old + 0.25 * Reward
    # w/ regret=lr/2 --> (1 - (lr + regret*(1-Reward))) * Value_old + (lr + regret*(1-Reward)) * Reward
    # w/ confirmation bias with cb=lr/2 
    self._q[non_chosen_action] += forget_update
    self._q[choice] += reward_update
    
    # Choice-Perseverance: Action-based updates
    self._c = np.zeros(self._n_actions)
    self._c[choice] += self._perseverance_bias
    # cpe = np.eye(self._n_actions)[choice] - self._c
    # self._c += self._alpha_perseverance * cpe

  @property
  def q(self):
    return (self._q + self._c)*self._beta
  
  def set_reward_prediction_error(self, update_rule: Callable):
    self._reward_prediction_error = update_rule


class AgentSindy:

  def __init__(
      self,
      sindy_models: Dict[str, ps.SINDy],
      n_actions: int=2,
      beta: Tuple[float]=(1., 1.),
      deterministic: bool=False,
      ):
    
    self._models = sindy_models
    self._q_init = 0.5
    self._deterministic = deterministic
    self._beta_reward = beta[0]
    self._beta_choice = beta[1]
    self._n_actions = n_actions
    self._prev_choice = None
    
    self._n_parameters = self._count_sindy_parameters()
    
    self.new_sess()

  def update(self, choice: int, reward: int):

      choice_repeat = np.max((0, 1 - np.abs(choice-self._prev_choice)))
      self._prev_choice = 0
      
      for action in range(self._n_actions):
        chosen = np.abs(choice - action) == 0
        
        # reward network
        if chosen:
          # current action was chosen
          if 'xLR' in self._models:
            learning_rate = self._models['xLR'].predict(np.array([0]), u=np.array([self._q[action], reward, 1-reward]).reshape(1, -1))[-1]
            reward_prediction_error = reward - self._q[action]
            self._q[action] += learning_rate * reward_prediction_error
          if 'xC' in self._models:
            self._c[action] = self._models['xC'].predict(np.array([self._c[action]]), u=np.array([choice_repeat]).reshape(1, -1))[-1]
        else:
          # current action was not chosen
          if 'xQf' in self._models:
            self._q[action] = self._models['xQf'].predict(np.array([self._q[action]]), u=np.array([0]).reshape(1, -1))[-1]
          if 'xCf' in self._models:
            self._c[action] = self._models['xCf'].predict(np.array([self._c[action]]), u=np.array([0]).reshape(1, -1))[-1]
        
      self._prev_choice += choice
      
  def new_sess(self):
    """Reset the agent for the beginning of a new session."""
    self._q = self._q_init + np.zeros(self._n_actions)
    self._c = np.zeros(self._n_actions)
    self._prev_choice = -1
    
  def get_choice_probs(self) -> np.ndarray:
    """Compute the choice probabilities as softmax over q."""
    decision_variable = np.exp(self.q)
    choice_probs = decision_variable / np.sum(decision_variable)
    return choice_probs

  def get_choice(self) -> int:
    """Sample a choice, given the agent's current internal state."""
    choice_probs = self.get_choice_probs()
    if self._deterministic:
      choice = np.argmax(choice_probs)
    else:
      choice = np.random.choice(self._n_actions, p=choice_probs)
    return choice
  
  def _count_sindy_parameters(self, without_self=False):
    """Count the SINDy coefficients across all sub-models which are not zero

    Returns:
        int: number of sindy coefficients which are not zero
    """
    
    n_parameters = 0
    
    for m in self._models:
      index_self = self._models[m].get_feature_names().index(m)
      coeff_self = self._models[m].coefficients()[0, index_self]
      coeffs = self._models[m].coefficients()[0]
      if without_self and coeff_self == 1 and np.sum(coeffs[coeffs != index_self]) == 0:
        # the values are passed through the sub-model without any processing. 
        # coeff_self = 1 and the rest is 0: 
        # Example: models['xC'](Input: 1234) -> Output: 1234
        # Sub-model won't be considered for SINDy-Coefficients.
        continue
      else:
        for c in coeffs:
          if c != 0:
            n_parameters += 1
     
    return n_parameters
    
  @property
  def q(self):
    return self._q * self._beta_reward + self._c * self._beta_choice


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
      deterministic: bool = False,
      ):
        """Initialize the agent network.

        Args:
            model: A PyTorch module representing the RNN architecture
            n_actions: number of permitted actions (default = 2)
        """
        
        self._deterministic = deterministic
        self._q_init = 0.5
        if device != model.device:
          model = model.to(device)
        if isinstance(model, RLRNN):
          self._model = RLRNN(model._n_actions, model._hidden_size, model.init_value, list(model.history.keys()), device=model.device).to(model.device)
          self._model.load_state_dict(model.state_dict())
        else:
          self._model = model
        self._model.eval()
        self._n_actions = n_actions
        
        # self._directed_exploration_bias = self._model._directed_exploration_bias.item()
        
        self.new_sess()

    def new_sess(self):
      """Reset the network for the beginning of a new session."""
      self._model.set_initial_state(batch_size=1)
      self._xs = torch.zeros((1, 2))-1
      self.set_state()

    def get_logit(self):
      """Return the value of the agent's current state."""
      logit = self._q * self._model._beta_reward.item() + self._c * self._model._beta_choice.item() # + self._u #* self._directed_exploration_bias
      return logit
    
    def get_choice_probs(self) -> np.ndarray:
      """Predict the choice probabilities as a softmax over output logits."""
      decision_variable = self.get_logit()
      choice_probs = np.exp(decision_variable) / np.sum(np.exp(decision_variable))
      return choice_probs

    def get_choice(self):
      """Sample choice."""
      choice_probs = self.get_choice_probs()
      if self._deterministic:
        return np.argmax(choice_probs)
      else:
        return np.random.choice(self._n_actions, p=choice_probs)

    def update(self, choice: float, reward: float):
      self._xs = torch.tensor([[choice, reward]], device=self._model.device)
      with torch.no_grad():
        self._model(self._xs, self._model.get_state())
      self.set_state()
    
    def set_state(self):
      self._q = self._model.get_state()[-3].cpu().numpy()[0, 0]
      self._c = self._model.get_state()[-2].cpu().numpy()[0, 0]

    @property
    def q(self):
      return self.get_logit()


################
# ENVIRONMENTS #
################


class Environment:
  
  def __init__(self):
    pass
  
  def step(self, choice):
    pass


class EnvironmentBanditsFlips(Environment):
  """Env for 2-armed bandit task with reward probs that flip in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.8,
      reward_prob_low: float = 0.2,
  ):
    
    super(EnvironmentBanditsFlips, self).__init__()
    
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
  

class EnvironmentBanditsSwitch(Environment):
  """Env for 2-armed bandit task with fixed sets of reward probs that switch in blocks."""

  def __init__(
      self,
      block_flip_prob: float = 0.02,
      reward_prob_high: float = 0.75,
      reward_prob_low: float = 0.25,
      reward_prob_middle: float = 0.5,
      **kwargs,
  ):
    
    super(EnvironmentBanditsSwitch, self).__init__()
    
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    self._reward_prob_middle = reward_prob_middle
    
    self._n_blocks = 7
    
    # Choose a random block to start in
    self._block = np.random.randint(self._n_blocks)
    
    # Set up the new block
    self.new_block()

  def new_block(self):
    """Switch the reward probabilities for a new block."""
    
    # Choose a new random block
    block = np.random.randint(0, self._n_blocks)
    while block == self._block:
      block = np.random.randint(0, self._n_blocks)
    self._block = block
    
    # Set the reward probabilites
    if self._block == 0:
      self._reward_probs = [self._reward_prob_high, self._reward_prob_low]
    elif self._block == 1:
      self._reward_probs = [self._reward_prob_middle, self._reward_prob_middle]
    elif self._block == 2:
      self._reward_probs = [self._reward_prob_low, self._reward_prob_high]
    elif self._block == 3:
      self._reward_probs = [self._reward_prob_low, self._reward_prob_middle]
    elif self._block == 4:
      self._reward_probs = [self._reward_prob_middle, self._reward_prob_high]
    elif self._block == 5:
      self._reward_probs = [self._reward_prob_middle, self._reward_prob_low]
    elif self._block == 6:
      self._reward_probs = [self._reward_prob_high, self._reward_prob_middle]
      
  def step(self, choice):
    """Step the model forward given chosen action."""
    # Choose the reward probability associated with agent's choice
    reward_prob_trial = self.reward_probs[choice]

    # Sample a reward with this probability
    reward = float(np.random.binomial(1, reward_prob_trial))

    # Check whether to flip the block
    if np.random.uniform() < self._block_flip_prob:
      self.new_block()

    # Return the reward
    return float(reward)

  @property
  def reward_probs(self) -> np.ndarray:
    return self._reward_probs.copy()

  @property
  def n_actions(self) -> int:
    return 2


class EnvironmentBanditsDrift(Environment):
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
    
    super(EnvironmentBanditsDrift, self).__init__()
    
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
      # non-correlated reward probabilities
      self._reward_probs += drift
    else:
      # in the case of correlated reward probabilities: adding to one option means substracting the same drift from the other
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
  reward_probabilities: np.ndarray
  q: np.ndarray
  n_trials: int
  

Agent = Union[AgentQ, AgentNetwork, AgentSindy]
# Environment = Union[EnvironmentBanditsFlips, EnvironmentBanditsDrift, EnvironmentBanditsSwitch]


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
    # Log environment reward probabilities and Q-Values
    reward_probs[trial] = environment.reward_probs
    qs[trial] = agent.q
    # First - agent makes a choice
    choice = agent.get_choice()
    # Second - environment computes a reward
    reward = environment.step(choice)
    # Log choice and reward
    choices[trial] = choice
    rewards[trial] = reward
    # Third - agent updates its believes based on chosen action and received reward
    agent.update(choice, reward)
    
  experiment = BanditSession(n_trials=n_trials,
                             choices=choices[:-1].astype(int),
                             rewards=rewards[:-1],
                             reward_probabilities=reward_probs[:-1],
                             q=qs[:-1])
  return experiment, choices.astype(int), rewards


def create_dataset(
  agent: Agent,
  environment: Environment,
  n_trials_per_session: int,
  n_sessions: int,
  sequence_length: int = None,
  stride: int = 1,
  sample_parameters: bool = False,
  device=torch.device('cpu'),
  verbose=False,
  ) -> tuple[DatasetRNN, list[BanditSession], list[dict[str, float]]]:
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
  parameter_list = []

  for session in range(n_sessions):
    if verbose:
      print(f'Running session {session+1}/{n_sessions}...')
    agent.new_sess(sample_parameters=sample_parameters)
    experiment, choices, rewards = run_experiment(agent, environment, n_trials_per_session)
    experiment_list.append(experiment)
    # one-hot encoding of choices
    choices = np.eye(agent._n_actions)[choices]
    xs[:, session] = np.concatenate((choices[:-1], rewards[:-1].reshape(-1, 1)), axis=-1)
    ys[:, session] = choices[1:]
    if isinstance(agent, AgentQ):
      # add current parameters to list
      parameter_list.append(
        {
          'beta': copy(agent._beta),
          'alpha': copy(agent._alpha),
          'alpha_penalty': copy(agent._alpha_penalty),
          'confirmation_bias': copy(agent._confirmation_bias),
          'forget_rate': copy(agent._forget_rate),
          'perseverance_bias': copy(agent._perseverance_bias),
        }
      )

  dataset = DatasetRNN(
    xs=np.swapaxes(xs, 0, 1), 
    ys=np.swapaxes(ys, 0, 1),
    sequence_length=sequence_length,
    stride=stride,
    device=device)
  return dataset, experiment_list, parameter_list


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
  Qs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  qs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  cs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  choice_probs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  
  agent.new_sess()
  
  for trial in range(experiment.choices.shape[0]):
    # track all states
    Qs[trial] = agent.q
    qs[trial] = agent._q
    cs[trial] = agent._c
    
    choice_probs[trial] = agent.get_choice_probs()
    agent.update(int(choices[trial]), float(rewards[trial]))
  
  return (Qs, qs, cs), choice_probs, agent


###############
# DIAGNOSTICS #
###############


def plot_session(
  chosen: np.ndarray,
  rewards: np.ndarray,
  timeseries: Tuple[np.ndarray],
  timeseries_name: str,
  labels: Optional[Tuple[str]] = None,
  title: str = '',
  x_label = 'Trials',
  fig_ax = None,
  compare=False,
  color=None,
  binary=False,
  x_axis_info=True,
  y_axis_info=True,
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
  
  # choose_high = choices == 1
  # choose_low = choices == 0
  rewarded = rewards > 0.5
  # y_high = np.max(timeseries) + 0.1
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
    # choices = np.expand_dims(choices, 0)
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

  # if choices.max() <= 1:
  #   # Rewarded high
  #   ax.scatter(
  #       np.argwhere(choose_high & rewarded),
  #       y_high * np.ones(np.sum(choose_high & rewarded)),
  #       color='green',
  #       marker=3)
  #   ax.scatter(
  #       np.argwhere(choose_high & rewarded),
  #       y_high * np.ones(np.sum(choose_high & rewarded)),
  #       color='green',
  #       marker='|')
  #   # Omission high
  #   ax.scatter(
  #       np.argwhere(choose_high & 1 - rewarded),
  #       y_high * np.ones(np.sum(choose_high & 1 - rewarded)),
  #       color='red',
  #       marker='|')

  #   # Rewarded low
  ax.scatter(
      np.argwhere(chosen & rewarded),
      y_low * np.ones(np.sum(chosen & rewarded)),
      color='green',
      marker='|')
  ax.scatter(
      np.argwhere(chosen & rewarded),
      y_low * np.ones(np.sum(chosen & rewarded)),
      color='green',
      marker=2)
  # Omission Low
  ax.scatter(
      np.argwhere(chosen & 1 - rewarded),
      y_low * np.ones(np.sum(chosen & 1 - rewarded)),
      color='red',
      marker='|')

  if x_axis_info:
    ax.set_xlabel(x_label)
  else:
    ax.set_xticks(np.linspace(1, len(timeseries), 5))
    ax.set_xticklabels(['']*5)
    
  if y_axis_info:
    ax.set_ylabel(timeseries_name)
  else:
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(['']*5)
    
  ax.set_title(title)
