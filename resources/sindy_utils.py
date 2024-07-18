import numpy as np
from torch.nn.functional import cross_entropy, mse_loss
import torch
from typing import Iterable, List, Dict, Tuple, Callable, Union
import matplotlib.pyplot as plt

import pysindy as ps

from resources.bandits import Environment, AgentNetwork, AgentSindy, get_update_dynamics
from resources.rnn import EnsembleRNN
from resources.rnn_utils import DatasetRNN


def make_sindy_data(
    dataset,
    agent,
    sessions=-1,
    ):

  # Get training data for SINDy
  # put all relevant signals in x_train

  if not isinstance(sessions, Iterable) and sessions == -1:
    # use all sessions
    sessions = np.arange(len(dataset))
  else:
    # use only the specified sessions
    sessions = np.array(sessions)
    
  n_control = 2
  
  choices = np.stack([dataset[i].choices for i in sessions], axis=0)
  rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
  qs = np.stack([dataset[i].q for i in sessions], axis=0)
  
  choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
  for sess in sessions:
    # one-hot encode choices
    choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
    
  # concatenate all qs values of one sessions along the trial dimension
  qs_all = np.concatenate([np.stack([np.expand_dims(qs_sess[:, i], axis=-1) for i in range(agent._n_actions)], axis=0) for qs_sess in qs], axis=0)
  c_all = np.concatenate([np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in choices_oh], axis=0)
  r_all = np.concatenate([np.stack([r_sess for _ in range(agent._n_actions)], axis=0) for r_sess in rewards], axis=0)
  
  # get observed dynamics
  x_train = qs_all
  feature_names = ['q']

  # get control
  control_names = []
  control = np.zeros((*x_train.shape[:-1], n_control))
  control[:, :, 0] = c_all
  control_names += ['c']
  control[:, :, n_control-1] = r_all
  control_names += ['r']
  
  feature_names += control_names
  
  print(f'Shape of Q-Values is: {x_train.shape}')
  print(f'Shape of control parameters is: {control.shape}')
  print(f'Feature names are: {feature_names}')
  
  # make x_train and control sequences instead of arrays
  x_train = [x_train_sess for x_train_sess in x_train]
  control = [control_sess for control_sess in control]
 
  return x_train, control, feature_names


def create_dataset(
  agent: AgentNetwork,
  data: Union[Environment, DatasetRNN, np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
  n_trials_per_session: int,
  n_sessions: int,
  normalize: bool = False,
  shuffle: bool = False,
  verbose: bool = False,
  trimming: int = 0,
  ):
  
  if not isinstance(data, Environment):
    if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 2:
      data = [data]
    if verbose:
      Warning('data is not of type Environment. Checking for correct number of sessions and trials per session with respect to the given data object.')
    if n_sessions > len(data):
      n_sessions = len(data)
    if not isinstance(data, DatasetRNN):
      if n_trials_per_session > data[0].shape[0]:
        n_trials_per_session = data[0].shape[0]
    else:
      if n_trials_per_session > data.xs.shape[1]:
        n_trials_per_session = data.xs.shape[1]    
  
  keys_x = [key for key in agent._model.history.keys() if key.startswith('x')]
  keys_c = [key for key in agent._model.history.keys() if key.startswith('c')]
  
  # x_train = np.zeros((n_sessions*agent._n_actions*(n_trials_per_session-1), 2, len(keys_x)))
  # control = np.zeros((n_sessions*agent._n_actions*(n_trials_per_session-1), 2, len(keys_c)))
  x_train = {key: [] for key in keys_x}
  control = {key: [] for key in keys_c}
  
  for session in range(n_sessions):
    agent.new_sess()
    
    for trial in range(n_trials_per_session):
      # generate trial data
      choice = agent.get_choice()
      if isinstance(data, Environment):
        reward = data.step(choice)
      elif isinstance(data, DatasetRNN):
        reward = data.xs[session, trial, -1].item()
      else:
        reward = data[session][trial, -1]
        if isinstance(reward, torch.Tensor):
          reward = reward.item()
        
      agent.update(choice, reward)
    
    # sort the data of one session into the corresponding signals
    for key in agent._model.history.keys():
      if len(agent._model.history[key]) > 1:
        # TODO: resolve ugly workaround with class distinction
        history = agent._model.history[key]
        if isinstance(agent._model, EnsembleRNN):
          history = history[-1]
        values = torch.concat(history).detach().cpu().numpy()[trimming:]
        if key in keys_x:
          # add values of interest of one session as trajectory
          for i_action in range(agent._n_actions):
            x_train[key] += [v for v in values[:, :, i_action]]
        elif key in keys_c:
          # add control signals of one session as corresponding trajectory
          if values.shape[-1] == 1:
            values = np.repeat(values, 2, -1)
          for i_action in range(agent._n_actions):
            control[key] += [v for v in values[:, :, i_action]]
              
  # get all keys of x_train and control that have no values and remove them
  keys_x = [key for key in keys_x if len(x_train[key]) > 0]
  keys_c = [key for key in keys_c if len(control[key]) > 0]
  x_train = {key: x_train[key] for key in keys_x}
  control = {key: control[key] for key in keys_c}
  feature_names = keys_x + keys_c
  
  # make x_train and control List[np.ndarray] with shape (n_trials_per_session-1, len(keys)) instead of dictionaries
  x_train_list = []
  control_list = []
  for i in range(len(control[keys_c[0]])):
    x_train_list.append(np.stack([x_train[key][i] for key in keys_x], axis=-1))
    control_list.append(np.stack([control[key][i] for key in keys_c], axis=-1))
  
  if normalize:
    index_cQr = keys_c.index('cQr') if 'cQr' in keys_c else None
    # compute scaling parameters
    x_max, x_min = np.max(np.stack(x_train_list)), np.min(np.stack(x_train_list))
    beta = x_max - x_min
    # normalize data (TODO: find better solution for cQr)
    for i in range(len(x_train_list)):
      x_train_list[i] = (x_train_list[i] - x_min) / beta
      if 'cQr' in keys_c:
        control_list[i][:, index_cQr] = control_list[i][:, index_cQr] / beta
  else:
    beta = 1
  
  if shuffle:
    shuffle_idx = np.random.permutation(len(x_train_list))
    x_train_list = [x_train_list[i] for i in shuffle_idx]
    control_list = [control_list[i] for i in shuffle_idx]
  
  return x_train_list, control_list, feature_names, beta


def optimize_beta(experiment, agent: AgentNetwork, agent_sindy: AgentSindy, plot=False):
  # fit beta parameter of softmax by fitting on choice probability of the RNN by simple grid search

  # number of observed points
  n_points = 100

  # get choice probabilities of the RNN
  _, choice_probs_rnn = get_update_dynamics(experiment, agent)

  # set prior for beta parameter; x_max seems to be a good starting point
  # beta_range = np.linspace(x_max-1, x_max+1, n_points)
  beta_range = np.linspace(1, 10, n_points)

  # get choice probabilities of the SINDy agent for each beta in beta_range
  choice_probs_sindy = np.zeros((len(beta_range), len(choice_probs_rnn), agent._n_actions))
  for i, beta in enumerate(beta_range):
      agent_sindy._beta = beta
      _, choice_probs_sindy_beta = get_update_dynamics(experiment, agent_sindy)
      
      # add choice probabilities to choice_probs_sindy
      choice_probs_sindy[i, :, :] = choice_probs_sindy_beta
      
  # get best beta value by minimizing the error between choice probabilities of the RNN and the SINDy agent
  errors = np.zeros(len(beta_range))
  for i in range(len(beta_range)):
      errors[i] = np.sum(np.abs(choice_probs_rnn - choice_probs_sindy[i]))

  # get right beta value
  beta = beta_range[np.argmin(errors)]

  if plot:
    # plot error plot with best beta value in title
    plt.plot(beta_range, errors)
    plt.title(f'Error plot with best beta={beta}')
    plt.xlabel('Beta')
    plt.ylabel('MAE')
    plt.show()

  return beta


def check_library_setup(library_setup: Dict[str, List[str]], feature_names: List[str], verbose=False) -> bool:
  msg = '\n'
  for key in library_setup.keys():
    if key not in feature_names:
      msg += f'Key {key} not in feature_names.\n'
    else:
      for feature in library_setup[key]:
        if feature not in feature_names:
          msg += f'Key {key}: Feature {feature} not in feature_names.\n'
  if msg != '\n':
    msg += f'Valid feature names are {feature_names}.\n'
    print(msg)
    return False
  else:
    if verbose:
      print('Library setup is valid. All keys and features appear in the provided list of features.')
    return True
  

def remove_control_features(control_variables: List[np.ndarray], feature_names: List[str], target_feature_names: List[str]) -> List[np.ndarray]:
  control_new = []
  for control in control_variables:
    remaining_control_variables = [control[:, feature_names.index(feature)] for feature in target_feature_names]
    if len(remaining_control_variables) > 0:
      control_new.append(np.stack(remaining_control_variables, axis=-1))
    else:
      control_new = None
      break
  return control_new


def conditional_filtering(x_train: List[np.ndarray], control: List[np.ndarray], feature_names: List[str], relevant_feature: str, condition: float, remove_relevant_feature=True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  x_train_relevant = []
  control_relevant = []
  x_features = [feature_names[0]]  #[feature for feature in feature_names if feature.startswith('x')]
  control_features = feature_names[1:]  #[feature for feature in feature_names if feature.startswith('c')]
  for i, x, c in zip(range(len(x_train)), x_train, control):
    if relevant_feature in feature_names:
      i_relevant = control_features.index(relevant_feature)
      if c[0, i_relevant] == condition:
        x_train_relevant.append(x)
        control_relevant.append(c)
        if remove_relevant_feature:
          control_relevant[-1] = np.delete(control_relevant[-1], i_relevant, axis=-1)
  
  if remove_relevant_feature:
    control_features.remove(relevant_feature)
    
  return x_train_relevant, control_relevant, x_features+control_features


def setup_library(library_setup: Dict[str, List[str]]) -> Dict[str, Tuple[ps.feature_library.base.BaseFeatureLibrary, List[str]]]:
  libraries = {key: None for key in library_setup.keys()}
  feature_names = {key: [key] + library_setup[key] for key in library_setup.keys()}
  for key in library_setup.keys():
      library = ps.PolynomialLibrary(degree=2)
      library.fit(np.random.rand(10, len(feature_names[key])))
      print(library.get_feature_names_out(feature_names[key]))
      libraries[key] = (library, feature_names[key])
  
  ps.ConcatLibrary([libraries[key][0] for key in libraries.keys()])
  
  return libraries


def constructor_update_rule_sindy(sindy_models):
  def update_rule_sindy(q, h, choice, reward, spillover_update):
      # mimic behavior of rnn with sindy
      
      blind_update, correlation_update, reward_update, action_update = 0, 0, 0, 0
      
      # action network
      if choice == 1 and 'xH' in sindy_models:
        action_update = sindy_models['xH'].predict(np.array([q]), u=np.array([choice]).reshape(1, -1))[-1] - q  # get only the difference between q and q_update as h is later added to q
      
      # value network      
      if choice == 1 and 'xQr' in sindy_models:
          # reward-based update for chosen action
          reward_update = sindy_models['xQr'].predict(np.array([q]), u=np.array([reward, 1-reward]).reshape(1, -1))[-1] - q
      
      if choice == 0 and 'xQf' in sindy_models:
          # blind update for non-chosen action
          blind_update = sindy_models['xQf'].predict(np.array([q]), u=np.array([0]).reshape(1, -1))[-1] - q
      
      if choice == 0 and 'xQc' in sindy_models:
          # correlation update for non-chosen action
          correlation_update = sindy_models['xQc'].predict(np.array([q+blind_update[0]]), u=np.array([spillover_update]).reshape(1, -1))[-1] - q
      
      return q+blind_update+correlation_update+reward_update, action_update
    
  return update_rule_sindy


def sindy_loss_x(agent_sindy: AgentSindy, x_data: DatasetRNN, loss_fn: Callable = cross_entropy):
  """Compute the loss of the SINDy model directly on the data in x-coordinates to get a better feeling for the effectivity of certain adjustments.
  This loss is not used for SINDy-Training, but for analysis purposes only.

  Args:
      model (ps.SINDy): _description_
      x_data (DatasetRNN): _description_
      loss_fn (Callable, optional): _description_. Defaults to cross_entropy.
  """
  
  loss = 0
  for x, y in x_data:
    agent_sindy.new_sess()
    qs = np.zeros_like(y)
    loss_session = 0
    for t in range(x.shape[1]):
      # get choice from agent for current state
      # choice_probs[t] = np.expand_dims(agent_sindy.get_choice_probs(), 0)
      qs[t] = agent_sindy.q.reshape(1, -1)
      # update state of agent
      action = np.argmax(x[t, :-1])
      reward = x[t, -1]
      agent_sindy.update(action, reward)
      # compute loss
      loss_session += loss_fn(y[t], torch.tensor(qs[t])).item()
    loss += loss_session/x.shape[1]
  loss /= len(x_data)
  
  return loss


def sindy_loss_z(agent_sindy: AgentSindy, x_data: DatasetRNN, agent_rnn: AgentNetwork = None, z_data: np.ndarray = None, loss_fn: Callable = mse_loss):
  """Compute the loss of the SINDy model on the data in z-coordinates.

  Args:
      agent_sindy (AgentSindy): _description_
      z_data (np.ndarray): _description_
      loss_fn (Callable, optional): _description_. Defaults to mse_loss.
  """
  
  if agent_rnn is None and z_data is None:
    raise ValueError('Either agent_rnn or z_data must be provided.')
  if agent_rnn is not None and z_data is not None:
    raise ValueError('Only one of agent_rnn or z_data must be provided.')
  
  if z_data is not None and len(x_data) != len(z_data):
    raise ValueError('Length of x_data and z_data must be equal.')
  
  loss = 0
  for i, data in enumerate(x_data):
    x, y = data
    # x, y = next(iter(x_data))
    if z_data is not None:
      z = z_data[i]
    else:
      z = torch.zeros_like(y)
    qs = np.zeros_like(y)
    loss_session = 0
    agent_sindy.new_sess()
    if agent_rnn is not None:
      agent_rnn.new_sess()
    
    for t in range(x.shape[0]):
      # get Q-value from rnn agent
      if agent_rnn is not None:
        z[t] = torch.tensor(agent_rnn.q)
      # get Q-Value from sindy agent
      qs[t] = agent_sindy.q * agent_sindy._beta
      # update state of agent
      action = np.argmax(x[t, :-1])
      reward = x[t, -1]
      agent_rnn.update(action, reward)
      agent_sindy.update(action, reward)
      # compute loss for current timestep
      # TODO: z and qs are differently scaled at the moment (SINDy produces values between 0 and 1)
      loss_session += loss_fn(z[t], torch.tensor(qs[t])).item()
    
    loss += loss_session/x.shape[1]
  loss /= len(x_data)
  
  return loss
    