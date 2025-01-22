import numpy as np
import torch
from sklearn.metrics import log_loss, mean_squared_error
from typing import Iterable, List, Dict, Tuple, Callable, Union
import matplotlib.pyplot as plt

import pysindy as ps

from resources.bandits import Environment, AgentNetwork, AgentSindy, get_update_dynamics, BanditSession
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
  data: Union[Environment, DatasetRNN, List[BanditSession], np.ndarray, torch.Tensor, List[np.ndarray], List[torch.Tensor]],
  n_trials_per_session: int,
  n_sessions: int,
  shuffle: bool = False,
  verbose: bool = False,
  trimming: bool = False,
  clear_offset: bool = False,
  ):
  
  highpass_threshold = 1e-2
  
  if not isinstance(data, Environment):
    if isinstance(data, (np.ndarray, torch.Tensor)) and data.ndim == 2:
      data = [data]
    if verbose:
      Warning('data is not of type Environment. Checking for correct number of sessions and trials per session with respect to the given data object.')
    if isinstance(data, DatasetRNN):
      if n_trials_per_session > data.xs.shape[1] or n_trials_per_session == -1 or n_trials_per_session == None:
        n_trials_per_session = data.xs.shape[1]
      if n_sessions > data.xs.shape[0] or n_sessions == -1 or n_sessions == None:
        n_trials_per_session = data.xs.shape[0]
    else:
        if isinstance(data[0], BanditSession):
          if n_trials_per_session == None or n_trials_per_session == -1 or n_trials_per_session > data[0].choices.shape[0]:
            n_trials_per_session = data[0].choices.shape[0]  
          if n_sessions == None or n_sessions == -1 or n_sessions > len(data):
            n_sessions = len(data)
        else:
          if n_trials_per_session > data[0].shape[0]: 
            n_trials_per_session = data[0].shape[0]
          
  keys_x = [key for key in agent._model.history.keys() if key.startswith('x_')]
  keys_c = [key for key in agent._model.history.keys() if key.startswith('c_')]
    
  x_train = {key: [] for key in keys_x}
  control = {key: [] for key in keys_c}
  
  for session in range(n_sessions):
    # perform agent updates to record values over trials
    if isinstance(data, Environment):
      agent.new_sess()
      for _ in range(n_trials_per_session):
        # generate trial data
        choice = agent.get_choice()
        reward = data.step(choice)
        agent.update(choice, reward)
    elif isinstance(data[0], BanditSession):
      # fill up history of rnn-agent
      _, _, agent = get_update_dynamics(data[session], agent)
    
    trimming = int(0.25*n_trials_per_session) if trimming else 0
    
    # sort the data of one session into the corresponding signals
    for key in agent._model.history.keys():
      if len(agent._model.history[key]) > 1:
        # get all recorded values for the current session of one specific key 
        history = agent._model.history[key]
        # create tensor from list of tensors 
        values = torch.concat(history).detach().cpu().numpy()[trimming:]
        if clear_offset and key in keys_x:
          values -= np.min(values)
        # remove insignificant updates with high-pass: check if dv/dt > threshold; otherwise set v(t=1) = v(t=0)
        dvdt = np.abs(np.diff(values, axis=1).reshape(values.shape[0], values.shape[2]))
        for i_action in range(values.shape[-1]):
          values[:, 1, i_action] = np.where(dvdt[:, i_action] > highpass_threshold, values[:, 1, i_action], values[:, 0, i_action])
        # in the case of 1D values along actions dim: Create 2D values by repeating along the actions dim (e.g. reward in non-counterfactual experiments) 
        if values.shape[-1] == 1:
            values = np.repeat(values, agent._n_actions, -1)
        # add values of interest of one session as trajectory
        for i_action in range(agent._n_actions):
          if key in keys_x:
            x_train[key] += [v for v in values[:, :, i_action]]
          elif key in keys_c:
            control[key] += [v for v in values[:, :, i_action]]
  
  feature_names = keys_x + keys_c
  
  # make arrays from dictionaries
  x_train_array = np.zeros((len(x_train[keys_x[0]]), 2, len(keys_x)))
  control_array = np.zeros((len(x_train[keys_x[0]]), 2, len(keys_c)))
  for i, key in enumerate(x_train):
    x_train_array[:, :, i] = np.stack(x_train[key])
  for i, key in enumerate(control):
    control_array[:, :, i] = np.stack(control[key])
  
  if shuffle:
    shuffle_idx = np.random.permutation(len(x_train_array))
    x_train_array = x_train_array[shuffle_idx]
    control_array = control_array[shuffle_idx]
  
  return x_train_array, control_array, feature_names


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


def sindy_loss_x(agent: Union[AgentSindy, AgentNetwork], data: List[BanditSession], loss_fn: Callable = log_loss):
  """Compute the loss of the SINDy model directly on the data in x-coordinates i.e. predicting behavior.
  This loss is not used for SINDy-Training, but for analysis purposes only.

  Args:
      model (ps.SINDy): _description_
      x_data (DatasetRNN): _description_
      loss_fn (Callable, optional): _description_. Defaults to log_loss.
  """
  
  loss_total = 0
  for experiment in data:
    agent.new_sess()
    choices = experiment.choices
    rewards = experiment.rewards
    loss_session = 0
    for t in range(len(choices)-1):
      beta = agent._beta if hasattr(agent, "_beta") else 1
      y_pred = np.exp(agent.q * beta)/np.sum(np.exp(agent.q * beta))
      agent.update(choices[t], rewards[t])
      loss_session += loss_fn(np.eye(agent._n_actions)[choices[t+1]], y_pred)
    loss_total += loss_session/(t+1)
  return loss_total/len(data)


def bandit_loss(agent: Union[AgentSindy, AgentNetwork], data: List[BanditSession], coordinates: str = "x"):
  """Compute the loss of the SINDy model directly on the data in z-coordinates i.e. predicting q-values.
  This loss is also used for SINDy-Training.

  Args:
      model (ps.SINDy): _description_
      x_data (DatasetRNN): _description_
      loss_fn (Callable): _description_. Defaults to log_loss.
      coordinates (str): Defines the coordinate system in which to compute the loss. Can be either "x" (predicting behavior) or "z" (comparing choice probabilities). Defaults to "x".
      """
  
  loss_total = 0
  for experiment in data:
    agent.new_sess()
    choices = experiment.choices
    rewards = experiment.rewards
    qs = np.exp(experiment.q)/np.sum(np.exp(experiment.q))
    loss_session = 0
    for t in range(len(choices)-1):
      beta = agent._beta if hasattr(agent, "_beta") else 1
      y_pred = np.exp(agent.q * beta)/np.sum(np.exp(agent.q * beta))
      agent.update(choices[t], rewards[t])
      if coordinates == 'x':
        y_target = np.eye(agent._n_actions)[choices[t+1]]
        loss_session = log_loss(y_target, y_pred)
      elif coordinates == 'z':
        y_target = qs[t]
        loss_session = mean_squared_error(y_target, y_pred)
    loss_total += loss_session/(t+1)
  return loss_total/len(data)
