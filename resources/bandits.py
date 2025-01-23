from typing import NamedTuple, Union, Optional, Dict, Callable, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from copy import copy
import torch
import torch.utils
import pysindy as ps

from resources.rnn import RLRNN
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
      beta_reward: float = 3.,
      alpha_reward: float = 0.2,
      alpha_penalty: float = -1,
      alpha_counterfactual: float = 0.,
      beta_choice: float = 0.,
      alpha_choice: float = 1.,
      forget_rate: float = 0.,
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
    
    self._list_params = ['beta_reward', 'alpha_reward', 'alpha_penalty', 'alpha_counterfactual', 'beta_choice', 'alpha_choice', 'confirmation_bias', 'forget_rate']
    
    self._mean_beta_reward = beta_reward
    self._mean_alpha_reward = alpha_reward
    self._mean_alpha_penalty = alpha_penalty if alpha_penalty >= 0 else alpha_reward
    self._mean_confirmation_bias = confirmation_bias
    self._mean_forget_rate = forget_rate
    self._mean_beta_choice = beta_choice
    self._mean_alpha_choice = alpha_choice
    self._mean_alpha_counterfactual = alpha_counterfactual
    
    self._beta_reward = beta_reward
    self._alpha_reward = alpha_reward
    self._alpha_penalty = alpha_penalty if alpha_penalty >= 0 else alpha_reward
    self._confirmation_bias = confirmation_bias
    self._forget_rate = forget_rate
    self._beta_choice = beta_choice
    self._alpha_choice = alpha_choice
    self._alpha_counterfactual = alpha_counterfactual
    
    self._n_actions = n_actions
    self._parameter_variance = self.check_parameter_variance(parameter_variance)
    self._q_init = 0.5
    
    self.new_sess()

    _check_in_0_1_range(alpha_reward, 'alpha')
    if alpha_penalty >= 0:
      _check_in_0_1_range(alpha_penalty, 'alpha_penalty')
    _check_in_0_1_range(alpha_counterfactual, 'alpha_countefactual')
    _check_in_0_1_range(alpha_choice, 'alpha_choice')
    _check_in_0_1_range(forget_rate, 'forget_rate')
    
    self._reward_prediction_error = lambda q, reward: reward-q
    
  def check_parameter_variance(self, parameter_variance):
    if isinstance(parameter_variance, float):
      par_var_dict = {}
      for key in self._list_params:
        par_var_dict[key] = parameter_variance
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
  
  def new_sess(self, sample_parameters=False, **kwargs):
    """Reset the agent for the beginning of a new session."""
    self._q = np.full(self._n_actions, self._q_init)
    self._c = np.zeros(self._n_actions)
    
    # sample new parameters
    if sample_parameters:
      sanity = False
      while not sanity:
        # sample new parameters until all sanity checks are passed
        self._beta_reward = np.clip(np.random.normal(self._mean_beta_reward, self._mean_beta_reward/2 if self._parameter_variance['beta_reward'] == -1 else self._parameter_variance['beta_reward']), 0, 2*self._mean_beta_reward)
        self._beta_choice = np.clip(np.random.normal(self._mean_beta_choice, self._mean_beta_choice/2 if self._parameter_variance['beta_choice'] == -1 else self._parameter_variance['beta_choice']), 0, 2*self._mean_beta_choice)
        self._alpha_reward = np.clip(np.random.normal(self._mean_alpha_reward, self._mean_alpha_reward/2 if self._parameter_variance['alpha_reward'] == -1 else self._parameter_variance['alpha_reward']), 0 , 1)
        self._alpha_penalty = np.clip(np.random.normal(self._mean_alpha_penalty, self._mean_alpha_penalty/2 if self._parameter_variance['alpha_penalty'] == -1 else self._parameter_variance['alpha_penalty']), 0, 1)
        self._alpha_choice = np.clip(np.random.normal(self._mean_alpha_choice, self._mean_alpha_choice/2 if self._parameter_variance['alpha_choice'] == -1 else self._parameter_variance['alpha_choice']), 0, 1)
        self._alpha_counterfactual = np.clip(np.random.normal(self._mean_alpha_counterfactual, self._mean_alpha_counterfactual/2 if self._parameter_variance['alpha_counterfactual'] == -1 else self._parameter_variance['alpha_counterfactual']), 0, 1)
        self._confirmation_bias = np.clip(np.random.normal(self._mean_confirmation_bias, self._mean_confirmation_bias/2 if self._parameter_variance['confirmation_bias'] == -1 else self._parameter_variance['confirmation_bias']), 0, 1)
        self._forget_rate = np.clip(np.random.normal(self._mean_forget_rate, self._mean_forget_rate/2 if self._parameter_variance['forget_rate'] == -1 else self._parameter_variance['forget_rate']), 0, 1)

        # sanity checks
        # 1. (alpha, alpha_penalty) + confirmation_bias*max_confirmation must be in range(0, 1)
        #     with max_confirmation = (q-q0)(r-q0) = +/- 0.25
        max_learning_rate = self._alpha_reward + self._confirmation_bias*0.25 <= 1 and self._alpha_penalty + self._confirmation_bias*0.25 <= 1
        min_learning_rate = self._alpha_reward + self._confirmation_bias*-0.25 >= 0 and self._alpha_penalty + self._confirmation_bias*-0.25 >= 0 
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

  def update(self, choice: int, reward: np.ndarray, *args):
    """Update the agent after one step of the task.

    Args:
      choice: The choice made by the agent. 0 or 1
      reward: The reward received by the agent. 0 or 1
    """
    
    # Reward-based updates
    non_chosen_action = np.arange(self._n_actions) != choice
    
    # if only factual feedback, assign reward to the correct column
    new_reward = np.zeros_like(reward)
    if reward[1] == -1:
      new_reward[choice] += reward[0]
      new_reward[non_chosen_action] += -1
    else:
      new_reward = reward
    
    # adjust learning rates for every received reward
    alpha = np.zeros_like(reward)
    for action in range(self._n_actions):
      if action == choice:
        # factual case
        alpha_r = self._alpha_reward
        alpha_p = self._alpha_penalty
      else: 
        # counterfactual case
        # counterfactual learning rate applies in case of foregone reward
        # no counterfactual learning in case of foregone penalty
        alpha_r = self._alpha_counterfactual
        alpha_p = 0
        
      # asymmetric learning rates
      alpha[action] = alpha_r if new_reward[action] > 0.5 else alpha_p
      
      if action == choice:
        # add confirmation bias to learning rate
        # Rollwage et al (2020): https://www.nature.com/articles/s41467-020-16278-6.pdf
        # if self._confirmation_bias:  
        # when any input to a cognitive mechanism is differentiable --> cognitive mechanism must be differentiable as well!
        # differentiable confirmation bias:
        alpha[action] += self._confirmation_bias * (self._q[action]-self._q_init)*(new_reward[action] - 0.5)
    
    # Reward-prediction-error
    rpe = self._reward_prediction_error(self._q, new_reward)
      
      # full learning rate equation w/ confirmation bias = 0.5: 
      # (xLR)[k+1]    = 0.25 + 0.5 * (Q-0.5) * (Reward - 0.5)
      #               = 0.25 + 0.5*Q*Reward - 0.25*Q - 0.25*Reward + 0.125
      # SINDy target: = 0.375 - 0.25*Q - 0.25*Reward + 0.5*Q*Reward

    reward_update = alpha * rpe
    
    # Forgetting - restore q-values of non-chosen actions towards the initial value
    forget_update = self._forget_rate * (self._q_init - self._q[non_chosen_action])
    
    # Value_new = Value_old - lr * Value_old + lr * Reward 
    # Standard (static) with lr=0.25 --> Value_new = (1-0.25) * Value_old + 0.25 * Reward
    # w/ regret=lr/2 --> (1 - (lr + regret*(1-Reward))) * Value_old + (lr + regret*(1-Reward)) * Reward
    # w/ confirmation bias with cb=lr/2 
    self._q[non_chosen_action] += reward_update[non_chosen_action] + forget_update
    self._q[choice] += reward_update[choice]
    
    # Choice-Perseverance: Action-based updates
    # self._c = np.zeros(self._n_actions)
    # self._c[choice] += self._perseverance_bias
    cpe = np.eye(self._n_actions)[choice] - self._c
    self._c += self._alpha_choice * cpe

  @property
  def q(self):
    return self._q*self._beta_reward + self._c*self._beta_choice
  
  def set_reward_prediction_error(self, update_rule: Callable):
    self._reward_prediction_error = update_rule


class AgentQ_SampleBetaDist(AgentQ):
  """An agent that runs simple Q-learning for the y-maze tasks.

  Attributes:
    alpha: The agent's learning rate
    beta: The agent's softmax temperature
    q: The agent's current estimate of the reward probability on each arm
  """

  def __init__(
      self,
      n_actions: int = 2,
      beta_reward: float = 3.,
      alpha_reward: float = 0.2,
      alpha_penalty: float = -1,
      alpha_counterfactual: float = 0.,
      beta_choice: float = 0.,
      alpha_choice: float = 1.,
      forget_rate: float = 0.,
      confirmation_bias: float = 0.,
      parameter_variance: Union[Dict[str, float], float] = 0.,
      ):
    
    super(AgentQ_SampleBetaDist, self).__init__(
      n_actions=n_actions,
      beta_reward=beta_reward,
      alpha_reward=alpha_reward,
      alpha_penalty=alpha_penalty,
      alpha_counterfactual=alpha_counterfactual,
      beta_choice=beta_choice,
      alpha_choice=alpha_choice,
      forget_rate=forget_rate,
      confirmation_bias=confirmation_bias,
      parameter_variance=parameter_variance,
      )
  
  def new_sess(self, sample_parameters=False, threshold=1e-1, **kwargs):
    """Reset the agent for the beginning of a new session."""
    self._q = np.full(self._n_actions, self._q_init)
    self._c = np.zeros(self._n_actions)
    
    # sample new parameters
    if sample_parameters:
      sanity_check = False
      while not sanity_check:
        # sample new parameters until all sanity checks are passed
        self._beta_reward = np.random.beta(0.5, 0.5) * 2 * self._mean_beta_reward
        self._beta_choice = np.random.beta(0.5, 0.5) * 2 * self._mean_beta_choice
        self._alpha_reward = np.random.beta(0.5, 0.5)
        self._alpha_penalty = np.random.beta(0.5, 0.5)
        self._alpha_choice = np.random.beta(0.5, 0.5)
        self._alpha_counterfactual = np.random.beta(0.5, 0.5)
        self._confirmation_bias = np.random.beta(0.5, 0.5)
        self._forget_rate = np.random.beta(0.5, 0.5)
        
        # Apply threshold to set variables to 0
        self._beta_reward = self._beta_reward if self._beta_reward > threshold else 0
        self._beta_choice = self._beta_choice if self._beta_choice > threshold else 0
        self._alpha_reward = self._alpha_reward if self._alpha_reward > threshold else 0
        self._alpha_penalty = self._alpha_penalty if self._alpha_penalty > threshold else 0
        self._alpha_choice = self._alpha_choice if self._alpha_choice > threshold else 0
        self._alpha_counterfactual = self._alpha_counterfactual if self._alpha_counterfactual > threshold else 0
        self._confirmation_bias = self._confirmation_bias if self._confirmation_bias > threshold else 0
        self._forget_rate = self._forget_rate if self._forget_rate > threshold else 0

        # sanity checks
        # 1. (alpha, alpha_penalty) + confirmation_bias*max_confirmation must be in range(0, 1)
        #     with max_confirmation = (q-q0)(r-q0) = +/- 0.25
        max_learning_rate = self._alpha_reward + self._confirmation_bias*0.25 <= 1 and self._alpha_penalty + self._confirmation_bias*0.25 <= 1
        min_learning_rate = self._alpha_reward + self._confirmation_bias*-0.25 >= 0 and self._alpha_penalty + self._confirmation_bias*-0.25 >= 0 
        sanity_check = max_learning_rate and min_learning_rate
    
    

class AgentSindy:

  def __init__(
      self,
      sindy_models: Dict[str, ps.SINDy],
      n_actions: int=2,
      beta_reward: float=1.,
      beta_choice: float=1.,
      deterministic: bool=False,
      counterfactual: bool=False,
      ):
    
    self._models = sindy_models
    self._q_init = 0.5
    self._deterministic = deterministic
    self._beta_reward = beta_reward
    self._beta_choice = beta_choice
    self._n_actions = n_actions
    self._counterfactual = counterfactual
    self._prev_choice = None    
    self._n_parameters = self._count_sindy_parameters()
    
    self.new_sess()

  def update(self, choice: int, reward: int, *args):

      choice_repeat = np.max((0, 1 - np.abs(choice-self._prev_choice)))
      self._prev_choice = 0

      for action in range(self._n_actions):
        reward_action = reward[action] if self._counterfactual else reward[0]
        
        # reward network
        if choice == action:
          # current action was chosen
          if 'x_V_LR' in self._models:
            learning_rate = self._models['x_V_LR'].predict(np.array([0]), u=np.array([self._q[action], reward_action]).reshape(1, -1))[-1]
            reward_prediction_error = reward_action - self._q[action]
            self._q[action] += learning_rate * reward_prediction_error
          if 'x_C' in self._models:
            self._c[action] = self._models['x_C'].predict(np.array([self._c[action]]), u=np.array([choice_repeat]).reshape(1, -1))[-1]
        else:
          # current action was not chosen
          if self._counterfactual and 'x_V_LR_cf' in self._models:
            learning_rate = self._models['x_V_LR_cf'].predict(np.array([0]), u=np.array([self._q[action], reward_action]).reshape(1, -1))[-1]
            reward_prediction_error = reward_action - self._q[action]
            self._q[action] += learning_rate * reward_prediction_error
          elif not self._counterfactual and 'x_V_nc' in self._models:
            self._q[action] = self._models['x_V_nc'].predict(np.array([self._q[action]]), u=np.array([0]).reshape(1, -1))[-1]
          if 'x_C_nc' in self._models:
            self._c[action] = self._models['x_C_nc'].predict(np.array([self._c[action]]), u=np.array([0]).reshape(1, -1))[-1]
      
      self._prev_choice += choice
      
  def new_sess(self, **kwargs):
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
      model: RLRNN,
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
        self._n_actions = n_actions

        self._model = RLRNN(
          n_actions=model._n_actions, 
          hidden_size=model._hidden_size, 
          n_participants=model._n_participants, 
          init_value=model.init_value, 
          list_sindy_signals=list(model.history.keys()), 
          counterfactual=model._counterfactual,
          device=device,
          )
        self._model.load_state_dict(model.state_dict())
        self._model = self._model.to(device)
        self._model.eval()
        
        self.new_sess()

    def new_sess(self, session: int = 0):
      """Reset the network for the beginning of a new session."""
      if not isinstance(session, torch.Tensor):
        session = torch.tensor(session, dtype=int)[None]
      
      self._model.set_initial_state(batch_size=1)
      
      self._xs = torch.zeros((1, self._n_actions*2+1))
      self._xs[0, -1] = session
      
      if self._model._n_participants > 0:
        participant_embedding = self._model.participant_embedding(session)
        self._beta_reward = self._model._beta_reward(participant_embedding).detach().cpu().item()
        self._beta_choice = self._model._beta_choice(participant_embedding).detach().cpu().item()
      else:
        self._beta_reward = self._model._beta_reward.detach().cpu().item()
        self._beta_choice = self._model._beta_choice.detach().cpu().item()
      
      self.set_state()

    def get_logit(self):
      """Return the value of the agent's current state."""
      logit = self._q * self._beta_reward + self._c * self._beta_choice
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

    def update(self, choice: float, reward: float, session: float):
      choice = torch.eye(self._n_actions)[int(choice)]
      self._xs = torch.concat([choice, torch.tensor(reward), torch.tensor(session)[None]])[None, None, :].to(device=self._model.device)
      with torch.no_grad():
        self._model(self._xs, self._model.get_state())
      self.set_state()
    
    def set_state(self):
      self._q = self._model.get_state()[-3].detach().cpu().numpy()[0, 0]
      self._c = self._model.get_state()[-2].detach().cpu().numpy()[0, 0]

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
      counterfactual: bool = False,
  ):
    
    super(EnvironmentBanditsFlips, self).__init__()
    
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    self._counterfactual = counterfactual
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

  def step(self, choice: int = None):
    """Step the model forward given chosen action."""
    # Choose the reward probability associated with agent's choice
    # reward_prob_trial = self.reward_probs[choice]

    # Sample a reward with this probability
    # reward = float(np.random.binomial(1, reward_prob_trial))

    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)
    
    # Check whether to flip the block
    if np.random.binomial(1, self._block_flip_prob):
      self.new_block()

    # Return the reward
    if self._counterfactual:
      return reward
    else:
      return reward[choice].reshape(1,)

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
      counterfactual: bool = False,
      **kwargs,
  ):
    
    super(EnvironmentBanditsSwitch, self).__init__()
    
    # Assign the input parameters as properties
    self._block_flip_prob = block_flip_prob
    self._reward_prob_high = reward_prob_high
    self._reward_prob_low = reward_prob_low
    self._reward_prob_middle = reward_prob_middle
    self._counterfactual = counterfactual
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
      
  def step(self, choice: int = None):
    """Step the model forward given chosen action."""
    # Choose the reward probability associated with agent's choice    
    # reward_prob_trial = self.reward_probs[choice]

    # Sample a reward with this probability
    # reward = float(np.random.binomial(1, reward_prob_trial))
    reward = np.array([float(np.random.binomial(1, prob)) for prob in self.reward_probs], dtype=float)

    # Check whether to flip the block
    if np.random.uniform() < self._block_flip_prob:
      self.new_block()

    # Return the reward
    if self._counterfactual:
      return reward
    else:
      return reward[choice].reshape(1,)

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
      counterfactual: bool = False,
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
    self._counterfactual = counterfactual

    # Sample new reward probabilities
    self._new_sess()

  def _new_sess(self):
    # Pick new reward probabilities.
    # Sample randomly between 0 and 1
    self._reward_probs = np.random.rand(self._n_actions)

  def step(self, choice: int) -> np.ndarray:
    """Run a single trial of the task.

    Args:
      choice: integer specifying choice made by the agent (must be less than
        n_actions.)

    Returns:
      reward: The reward to be given to the agent. 0 or 1.

    """
    # Check inputs
    # if choice not in range(self._n_actions):
    #   msg = (
    #       f'Found value for choice of {choice}, but must be in '
    #       f'{list(range(self._n_actions))}')
    #   raise ValueError(msg)

    # Sample reward with the probability of the chosen side
    # reward = np.random.rand() < self._reward_probs[choice]
    if not self._non_binary_reward:
      reward = np.array([float(np.random.rand() < self._reward_probs[i]) for i in range(self._n_actions)])
    else:
      reward = self._reward_probs#[choice]
    
    # if self._counterfactual:
    # reward = rewards[choice]
    # reward_c = rewards[np.arange(rewards.shape[0]) != choice]
    # else:
    #   reward = rewards[choice]
    #   reward_c = None
    
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

    # Return the reward
    if self._counterfactual:
      return reward
    else:
      return reward[choice].reshape(1,)

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
  session: np.ndarray
  reward_probabilities: np.ndarray
  q: np.ndarray
  n_trials: int
  
  def set_session(self, session: int):
    return self(choices=self.choices, rewards=self.rewards, session=np.full_like(self.session, session), reward_probabilities=self.reward_probabilities, q=self.q, n_trials=self.n_trials)
  
  def __getitem__(self, val):
    return self._replace(choices=self.choices.__getitem__(val), rewards=self.rewards.__getitem__(val), session=self.session.__getitem__(val), reward_probabilities=self.reward_probabilities.__getitem__(val), q=self.q.__getitem__(val), n_trials=self.choices.__getitem__(val).shape[0])
  
Agent = Union[AgentQ, AgentNetwork, AgentSindy]
# Environment = Union[EnvironmentBanditsFlips, EnvironmentBanditsDrift, EnvironmentBanditsSwitch]


###############
#  EXECUTION  #
###############


def run_experiment(
  agent: Agent,
  environment: Environment,
  n_trials: int,
  session_id: int = 0,
  ) -> BanditSession:
  """Runs a behavioral session from a given agent and environment.

  Args:
    agent: An agent object
    environment: An environment object
    n_trials: The number of steps in the session you'd like to generate

  Returns:
    experiment: A BanditSession holding choices and rewards from the session
  """
  
  choices = np.zeros(n_trials+1) - 1
  rewards = np.zeros((n_trials+1, environment.n_actions)) - 1
  qs = np.zeros((n_trials+1, environment.n_actions)) - 1
  reward_probs = np.zeros((n_trials+1, environment.n_actions)) - 1

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
    rewards[trial, :len(reward)] = reward
    
    # Third - agent updates its believes based on chosen action and received reward
    agent.update(choice, rewards[trial])
    
  experiment = BanditSession(n_trials=n_trials,
                             choices=choices[:-1].astype(int),
                             rewards=rewards[:-1],
                             session=np.full(rewards[:-1].shape[0], session_id).astype(int),
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
  
  xs = np.zeros((n_sessions, n_trials_per_session, agent._n_actions*2 + 1))
  ys = np.zeros((n_sessions, n_trials_per_session, agent._n_actions))
  experiment_list = []
  parameter_list = []

  for session in range(n_sessions):
    if verbose:
      print(f'Running session {session+1}/{n_sessions}...')
    agent.new_sess(sample_parameters=sample_parameters)
    experiment, choices, rewards = run_experiment(agent, environment, n_trials_per_session, session)
    experiment_list.append(experiment)
    
    # one-hot encoding of choices
    choices = np.eye(agent._n_actions)[choices]
    ys[session] = choices[1:]
    xs[session] = np.concatenate((choices[:-1], rewards[:-1], experiment.session[:, None]), axis=-1)
    
    if isinstance(agent, AgentQ):
      # add current parameters to list
      parameter_list.append(
        {
          'beta_reward': copy(agent._beta_reward),
          'alpha_reward': copy(agent._alpha_reward),
          'alpha_penalty': copy(agent._alpha_penalty),
          'beta_choice': copy(agent._beta_choice),
          'alpha_choice': copy(agent._alpha_choice),
          'confirmation_bias': copy(agent._confirmation_bias),
          'forget_rate': copy(agent._forget_rate),
        }
      )

  dataset = DatasetRNN(
    xs=xs, 
    ys=ys,
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
  # rewards = np.expand_dims(experiment.rewards, 1)
  rewards = experiment.rewards
  Qs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  qs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  cs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  choice_probs = np.zeros((experiment.choices.shape[0], agent._n_actions))
  
  agent.new_sess(session=experiment.session[0])
  
  for trial in range(experiment.choices.shape[0]):
    # track all states
    Qs[trial] = agent.q
    qs[trial] = agent._q# * agent._beta_reward
    cs[trial] = agent._c# * agent._beta_choice
    
    choice_probs[trial] = agent.get_choice_probs()
    agent.update(int(choices[trial]), rewards[trial], int(experiment.session[0]))
  
  return (Qs, qs, cs), choice_probs, agent


###############
# DIAGNOSTICS #
###############


def plot_session(
  choices: np.ndarray,
  rewards: np.ndarray,
  timeseries: Tuple[np.ndarray],
  timeseries_name: str,
  labels: Optional[Tuple[str]] = None,
  title: str = '',
  x_label = 'Trials',
  fig_ax = None,
  compare=False,
  color=None,
  x_axis_info=True,
  y_axis_info=True,
  ):
  """Plot data from a single behavioral session of the bandit task.

  Args:
    choices: The choices made by the agent
    rewards: The rewards received by the agent
    timeseries: The dynamical value of interest on each arm
    timeseries_name: The name of the timeseries
    labels: The labels for the lines in the plot
    fig_ax: A tuple of a matplotlib figure and axis to plot on
    compare: If True, plot multiple timeseries on the same plot
    color: A list of colors to use for the plot; at least as long as the number of timeseries
  """

  if color == None:
    color = [None]*len(timeseries)
  
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
  
  # Plot ticks relating to whether the option was chosen (factual) or not (counterfactual) and whether it was rewarded
  min_y, max_y = np.min(timeseries), np.max(timeseries)
  # diff_min_max = np.max((5e-2, max_y - min_y))  # make sure the difference is not <= 0
  diff_min_max = np.max((1e-1, max_y - min_y))
  
  x = np.arange(len(choices))
  chosen_y = min_y - 1e-1  # Lower position for chosen (bigger tick)
  not_chosen_y = max_y + 1e-1  # Slightly lower for not chosen (smaller tick)
  # not_chosen_y = chosen_y - 1e-1 * diff_min_max  # Slightly lower for not chosen (smaller tick)

  # Plot ticks for chosen options
  ax.scatter(x[(choices == 1) & (rewards == 1)], np.full(sum((choices == 1) & (rewards == 1)), chosen_y), color='green', s=100, marker='|')  # Large green tick for chosen reward
  ax.scatter(x[(choices == 1) & (rewards == 0)], np.full(sum((choices == 1) & (rewards == 0)), chosen_y), color='red', s=80, marker='|')  # Large red tick for chosen penalty

  # Plot ticks for not chosen options
  ax.scatter(x[(choices == 0) & (rewards == 1)], np.full(sum((choices == 0) & (rewards == 1)), not_chosen_y), color='green', s=100, marker='|')  # Small green tick
  ax.scatter(x[(choices == 0) & (rewards == 0)], np.full(sum((choices == 0) & (rewards == 0)), not_chosen_y), color='red', s=80, marker='|')  # Small red tick

  # ax.set_ylim(not_chosen_y, np.max((-not_chosen_y, np.max(timeseries + 1e-1 * diff_min_max))))
  
  if x_axis_info:
    ax.set_xlabel(x_label)
  else:
    # ax.set_xticks(np.linspace(1, len(timeseries), 5))
    ax.set_xticklabels(['']*5)
    
  if y_axis_info:
    ax.set_ylabel(timeseries_name)
  else:
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels(['']*5)
    
  ax.set_title(title)
