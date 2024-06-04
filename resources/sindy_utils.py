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


def get_q(experiment: BanditSession, agent: Union[AgentQ, AgentNetwork, AgentSindy]):
  """Compute Q-Values of a specific agent for a specific experiment.

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


def parse_equation_for_sympy(eq):
    # replace all blank spaces with '*' where necessary
    # only between number and letter in exactly this order
    blanks = [i for i, ltr in enumerate(eq) if ltr == ' ']
    for blank in blanks:
        if (eq[blank+1].isalpha() or eq[blank-1].isdigit()) and (eq[blank+1].isalpha() or eq[blank+1].isdigit()):
            eq = eq[:blank] + '*' + eq[blank+1:]
    
    # replace all '^' with '**'
    eq = eq.replace('^', '**')
    
    # remove all [k]
    eq = eq.replace('[k]', '')

    return eq
  
  
def make_sindy_data(
    experiment_list: List[BanditSession],
    agent: Union[AgentQ, AgentNetwork],
    sessions=-1,
    ):

  # Get training data for SINDy
  # put all training signals in x_train and control signals in control

  if sessions == -1:
    # use all sessions
    sessions = np.arange(len(experiment_list))
  elif not isinstance(sessions, Iterable):
    # use only the specified sessions
    sessions = np.array(sessions)
    
  n_control = 3  # reward and choice and last choice
  
  choices = np.stack([experiment_list[i].choices for i in sessions], axis=0)
  rewards = np.stack([experiment_list[i].rewards for i in sessions], axis=0)
  # qs = np.stack([experiment_list[i].q for i in sessions], axis=0)
  qs = []
  feature_names = []
  for key in experiment_list[0].q.keys():
    qs.append(np.stack([experiment_list[i].q[key] for i in sessions], axis=0))
    feature_names.append(key)
    
  choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
  for sess in sessions:
    # one-hot encode choices
    choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
  
  # create array with last choices
  last_choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
  last_choices_oh[:, 1:] = choices_oh[:, :-1]
  
  # concatenate all x_train signals of one session along the trial dimension
  x_train = []
  for x in qs:
    x_train.append(np.concatenate([np.stack([np.expand_dims(x_sess[:, i], axis=-1) for i in range(agent._n_actions)], axis=0) for x_sess in x], axis=0))
  choices = np.concatenate([np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in choices_oh], axis=0)
  last_choices = np.concatenate([np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in last_choices_oh], axis=0)
  rewards = np.concatenate([np.stack([r_sess for _ in range(agent._n_actions)], axis=0) for r_sess in rewards], axis=0)

  # put all value signals in one array
  x_train = np.concatenate(x_train, axis=-1)
  
  # put all control signals into one array
  control_names = []
  control = np.zeros((*x_train.shape[:-1], n_control))
  control[:, :, 0] = choices
  control_names += ['c']
  control[:, :, 1] = last_choices
  control_names += ['c[k-1]']
  control[:, :, n_control-1] = rewards
  control_names += ['r']
  
  feature_names += control_names
  
  print(f'Shape of training variables: {x_train.shape}')
  print(f'Shape of control parameters: {control.shape}')
  print(f'Feature names are: {feature_names}')
  
  # make x_train and control sequences instead of arrays
  x_train = [x_train_sess for x_train_sess in x_train]
  control = [control_sess for control_sess in control]
 
  return x_train, control, feature_names


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
  x_train_signals = np.zeros((n_sessions*agent._n_actions, n_trials_per_session, len(keys_x)))
  control_signals = np.zeros((n_sessions*agent._n_actions, n_trials_per_session, len(keys_c)))
  
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
          x_train_signals[2*session:2*session+2, trial, i_key] = signal
        elif key in keys_c:
          i_key = keys_c.index(key)
          control_signals[2*session:2*session+2, trial, i_key] = signal
        else:
          raise ValueError(f'Cannot sort key {key} into x_train (must start with x) or control signals (must start with c).')  
  
  # transform signals to lists
  x_train = [x for x in x_train_signals]
  control = [c for c in control_signals]
  
  return x_train, control, feature_names