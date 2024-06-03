import numpy as np
from typing import Union, Iterable, List

from bandits import *


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
    
  n_control = 2  # reward and choice
  
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
    
  # concatenate all x_train signals of one session along the trial dimension
  x_train = []
  for x in qs:
    x_train.append(np.concatenate([np.stack([np.expand_dims(x_sess[:, i], axis=-1) for i in range(agent._n_actions)], axis=0) for x_sess in x], axis=0))
  c_all = np.concatenate([np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in choices_oh], axis=0)
  r_all = np.concatenate([np.stack([r_sess for _ in range(agent._n_actions)], axis=0) for r_sess in rewards], axis=0)

  # put all x_train signals in one array
  x_train = np.concatenate(x_train, axis=-1)
  
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
