#@title Import libraries
import sys
import warnings

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import rnn, rnn_training, bandits, sindy_utils

# train model
train = True
checkpoint = False  # TODO: does not work somehow... # only relevant if train is True --> Determines whether to load trained parameters and continue training or start new training
data = False

path_data = 'data/dataset_train.pkl'
params_path = 'params/params_rnn_forget_f0_b5.pkl'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not data:
  # agent parameters
  agent_kw = 'basic'  #@param ['basic', 'quad_q'] 
  gen_alpha = .25 #@param
  gen_beta = 5 #@param
  forgetting_rate = 0.2 #@param
  perseveration_bias = 0.  #@param
  # environment parameters
  non_binary_reward = False #@param
  n_actions = 2 #@param
  sigma = .1  #@param

  # experiement parameters
  n_trials_per_session = 200  #@param
  n_sessions = 220  #@param

  # setup
  environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_rewards=non_binary_reward)
  agent = bandits.AgentQ(gen_alpha, gen_beta, n_actions, forgetting_rate, perseveration_bias)  

  dataset_train, experiment_list_train = bandits.create_dataset(
      agent=agent,
      environment=environment,
      n_trials_per_session=n_trials_per_session,
      n_sessions=n_sessions)

  dataset_test, experiment_list_test = bandits.create_dataset(
      agent=agent,
      environment=environment,
      n_trials_per_session=n_trials_per_session,
      n_sessions=n_sessions)
else:
  # load data
  with open(path_data, 'rb') as f:
      dataset_train = pickle.load(f)

# define model
model = rnn.RNN(
    n_actions=2, 
    hidden_size=16, 
    init_value=0.5,
    last_output=True,
    last_state=False,
    use_habit=False,
    ).to(device)

optimizer_rnn = torch.optim.Adam(model.parameters(), lr=1e-3)

if train:
  if checkpoint:
    # load trained parameters
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    optimizer_rnn.load_state_dict(state_dict['optimizer'])
    print('Loaded parameters.')

  #@title Fit the hybrid RNN
  print('Training the hybrid RNN...')
  model, optimizer_rnn, _ = rnn_training.fit_model(
      model=model,
      dataset=dataset_train,
      optimizer=optimizer_rnn,
      convergence_threshold=1e-5,
      epochs=10000,
      batch_size=None,
  )

  # save trained parameters  
  state_dict = {
    'model': model.state_dict(),
    'optimizer': optimizer_rnn.state_dict(),
  }
  torch.save(state_dict, params_path)

else:
  # load trained parameters
  state_dict = torch.load(params_path)
  model.load_state_dict(state_dict['model'])
  print(f'Loaded parameters from file {params_path}.')
  
# Synthesize a dataset using the fitted network
environment = bandits.EnvironmentBanditsDrift(0.1)
rnn_agent = bandits.AgentNetwork(model, n_actions=2, habit=False)
# dataset_rnn, experiment_list_rnn = bandits.create_dataset(rnn_agent, environment, 220, 10)

# bandits.plot_session(    compare=False,
#     choices=experiment_list_rnn[0].choices,
#     rewards=experiment_list_rnn[0].rewards,
#     timeseries=experiment_list_rnn[0].q[:, 0].reshape(-1, 1),
#     timeseries_name='Q Values of RNN data',
#     # labels=labels,
#     # color=colors,
#     binary=True,
#     )
# plt.show()

# bandits.plot_session(    compare=False,
#     choices=experiment_list_rnn[0].choices,
#     rewards=experiment_list_rnn[0].rewards,
#     timeseries=experiment_list_rnn[0].timeseries[:, 0].reshape(-1, 1),
#     timeseries_name='Choice Probs of RNN data',
#     # labels=labels,
#     # color=colors,
#     binary=True,
#     )
# plt.show()


# Analysis
session_id = 0

choices = experiment_list_test[session_id].choices
rewards = experiment_list_test[session_id].rewards

list_probs = []
list_qs = []

# get q-values from groundtruth
qs_test, probs_test = sindy_utils.get_q(experiment_list_test[session_id], agent)
list_probs.append(np.expand_dims(probs_test, 0))
list_qs.append(np.expand_dims(qs_test, 0))

# get q-values from trained rnn
qs_rnn, probs_rnn = sindy_utils.get_q(experiment_list_test[session_id], rnn_agent)
list_probs.append(np.expand_dims(probs_rnn, 0))
list_qs.append(np.expand_dims(qs_rnn, 0))

colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

# concatenate all choice probs and q-values
probs = np.concatenate(list_probs, axis=0)
qs = np.concatenate(list_qs, axis=0)

# normalize q-values
qs = (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=probs[:, :, 0],
    timeseries_name='Choice Probabilities',
    # labels=labels,
    color=colors,
    binary=not non_binary_reward,
    )
plt.show()

bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=qs[:, :, 0],
    timeseries_name='Q-Values',
    # labels=labels,
    title='Left arm',
    color=colors,
    binary=not non_binary_reward,
    )
plt.show()

bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=qs[:, :, 1],
    timeseries_name='Q-Values',
    # labels=labels,
    title='Right arm',
    color=colors,
    binary=not non_binary_reward,
    )
plt.show()

