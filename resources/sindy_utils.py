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


def create_dataset(
  agent: Agent,
  environment: Environment,
  n_trials_per_session: int,
  n_sessions: int,
  ):
  
  x_train = []
  control = []
  
  keys_x = [key for key in agent._model.history.keys() if key.startswith('x')]
  keys_c = [key for key in agent._model.history.keys() if key.startswith('c')]
  feature_names = keys_x + keys_c
  
  # x_train_signals = {key: [] for key in keys_x}
  # control_signals = {key: [] for key in keys_c}
  x_train = np.zeros((n_sessions*agent._n_actions, n_trials_per_session, len(keys_x)))
  control = np.zeros((n_sessions*agent._n_actions, n_trials_per_session, len(keys_c)))
  
  for session in range(n_sessions):
    agent.new_sess()
    
    for trial in range(n_trials_per_session):
      choice = agent.get_choice()
      reward = environment.step(choice)
      agent.update(choice, reward)
    
      # after each trial sort the collected history either into x_train or control signals
      for key in agent._model.history.keys():
        signal = agent._model.history[key][trial].reshape(-1)
        if key in keys_x:
          # get index of key in keys_x
          i_key = keys_x.index(key)
          x_train[2*session:2*session+2, trial, i_key] = signal
        elif key in keys_c:
          i_key = keys_c.index(key)
          control[2*session:2*session+2, trial, i_key] = signal
        else:
          raise ValueError(f'Cannot sort key {key} into x_train (must start with x) or control signals (must start with c).')  
  
  # transform signals to lists
  x_train = [x for x in x_train]
  control = [c for c in control]
  
  return x_train, control, feature_names