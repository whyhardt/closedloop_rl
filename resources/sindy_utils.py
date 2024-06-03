import numpy as np
from typing import Union, Iterable

from . import bandits

def get_q(experiment: bandits.BanditSession, agent: Union[bandits.AgentQ, bandits.AgentNetwork, bandits.AgentSindy]):
  """Compute Q-Values of a specific agent for a specific experiment.

  Args:
      experiment (bandits.BanditSession): _description_
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
    dataset,
    agent: bandits.AgentQ,
    sessions=-1,
    get_choices=True,
    # keep_sessions=False,
    ):

  # Get training data for SINDy
  # put all relevant signals in x_train

  if not isinstance(sessions, Iterable) and sessions == -1:
    # use all sessions
    sessions = np.arange(len(dataset))
  else:
    # use only the specified sessions
    sessions = np.array(sessions)
    
  if get_choices:
    n_control = 2
  else:
    n_control = 1
  
  # if keep_sessions:
  #   # concatenate all sessions along the trial dimensinon -> shape: (n_trials, n_sessions, n_features)
  #   choices = np.expand_dims(np.stack([dataset[i].choices for i in sessions], axis=1), -1)
  #   rewards = np.expand_dims(np.stack([dataset[i].rewards for i in sessions], axis=1), -1)
  #   qs = np.stack([dataset[i].q for i in sessions], axis=1)
  # else:
  # concatenate all sessions along the trial dimensinon -> shape: (n_trials*n_sessions, n_features)
  # choices = np.expand_dims(np.concatenate([dataset[i].choices for i in sessions], axis=0), -1)
  # rewards = np.expand_dims(np.concatenate([dataset[i].rewards for i in sessions], axis=0), -1)
  # qs = np.concatenate([dataset[i].q for i in sessions], axis=0)
  
  choices = np.stack([dataset[i].choices for i in sessions], axis=0)
  rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
  qs = np.stack([dataset[i].q for i in sessions], axis=0)
  
  if not get_choices:
    raise NotImplementedError('Only get_choices=True is implemented right now.')
    n_sessions = qs.shape[0]
    n_trials = qs.shape[1]*qs.shape[2]
    qs_all = np.zeros((n_sessions, n_trials))
    r_all = np.zeros((n_sessions, n_trials))
    c_all = None
    # concatenate the data of all arms into one array for more training data
    index_end_last_arm = 0
    for index_arm in range(agent._n_actions):
      index = np.where(choices==index_arm)[0]
      r_all[index_end_last_arm:index_end_last_arm+len(index)] = rewards[index]
      qs_all[index_end_last_arm:index_end_last_arm+len(index)] = qs[index, index_arm].reshape(-1, 1)
      index_end_last_arm += len(index)
  else:
    choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
    for sess in sessions:
      # one-hot encode choices
      choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
      # add choices as control parameter; no sorting required then
      # qs_all = np.concatenate([qs[sess, :, i] for i in range(agent._n_actions)], axis=1)
      # c_all = np.concatenate([choices[:, sess, i] for i in range(agent._n_actions)], axis=1)
      # r_all = np.concatenate([rewards for _ in range(agent._n_actions)], axis=1)
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
  if get_choices:
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
