import numpy as np
from typing import Union, Iterable, List

import pysindy as ps

from bandits import *


custom_library = {
  'functions': [
    # sub-library which is always included    
    lambda q,c,r: q,
    lambda q,c,r: r,
    lambda q,c,r: np.power(q, 2),
    lambda q,c,r: q*r,
    lambda q,c,r: np.power(r, 2),
    # sub-library if the possible action was chosen
    # lambda q,c,r: c,
    lambda q,c,r: c*q,
    lambda q,c,r: c*r,
    lambda q,c,r: c*np.power(q, 2),
    lambda q,c,r: c*q*r,
    lambda q,c,r: c*np.power(r, 2),
],
  'names': [
    # part library which is always included
    lambda q,c,r: f'{q}',
    lambda q,c,r: f'{r}',
    lambda q,c,r: f'{q}^2',
    lambda q,c,r: f'{q}*{r}',
    lambda q,c,r: f'{r}^2',
    # part library if the possible action was chosen
    # lambda q,c,r: f'{c}',
    lambda q,c,r: f'{c}*{q}',
    lambda q,c,r: f'{c}*{r}',
    lambda q,c,r: f'{c}*{q}^2',
    lambda q,c,r: f'{c}*{q}*{r}',
    lambda q,c,r: f'{c}*{r}^2',
],
}

custom_library_ps = ps.CustomLibrary(
    library_functions=custom_library['functions'],
    function_names=custom_library['names'],
    include_bias=True,
)


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
  environment: Environment,
  n_trials_per_session: int,
  n_sessions: int,
  ):
  
  keys_x = [key for key in agent._model.history.keys() if key.startswith('x')]
  keys_c = [key for key in agent._model.history.keys() if key.startswith('c')]
  
  x_train = np.zeros((n_sessions*agent._n_actions, n_trials_per_session-1, len(keys_x)))
  control = np.zeros((n_sessions*agent._n_actions, n_trials_per_session-1, len(keys_c)))
  
  for session in range(0, n_sessions, agent._n_actions):
    agent.new_sess()
    
    for trial in range(n_trials_per_session):
      # generate trial data
      choice = agent.get_choice()
      reward = environment.step(choice)
      agent.update(choice, reward)
    
    # sort the data of one session into the corresponding signals
    for key in agent._model.history.keys():
      if len(agent._model.history[key]) > 1:
        if key in keys_x:
          # add values of interest of one session as trajectory
          x_train_values = np.concatenate(agent._model.history[key][1:-1])
          i_key = keys_x.index(key)
          for i_action in range(agent._n_actions):
            x_train[session+i_action, :, i_key] = x_train_values[:, i_action]

        if key in keys_c:
          # add control signals of one session as corresponding trajectory
          control_values = np.concatenate(agent._model.history[key][1:])
          i_key = keys_c.index(key)
          if control_values.shape[-1] == 1:
            control_values = np.repeat(control_values, 2, -1)
          for i_action in range(agent._n_actions):
            control[session+i_action, :, i_key] = control_values[:, i_action]
        
  x_train = [x_session for x_session in x_train]
  control = [c_session for c_session in control]
  feature_names = keys_x + keys_c
  
  return x_train, control, feature_names
