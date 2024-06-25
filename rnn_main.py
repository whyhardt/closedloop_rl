#@title Import libraries
import sys
import warnings

import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import rnn, rnn_training, bandits, rnn_utils, sindy_utils, sindy_training

# train model
train = True
checkpoint = False
data = False

use_lstm = False

path_data = 'data/dataset_train.pkl'
params_path = 'params/params_lstm_b3.pkl'  # overwritten if data is False (adapted to the ground truth model)

# rnn parameters
hidden_size = 4
last_output = False
last_state = False

# ensemble parameters
sampling_replacement = True
n_submodels = 1
ensemble = False
voting_type = rnn.EnsembleRNN.MEDIAN  # necessary if ensemble==True

# training parameters
epochs = 10000
n_steps_per_call = 16  # None for full sequence
batch_size = 256  # None for one batch per epoch
learning_rate = 1e-2
convergence_threshold = 1e-6

# tracked variables in the RNN
x_train_list = ['xQf','xQr', 'xH']
control_list = ['ca','ca[k-1]', 'cr']
sindy_feature_list = x_train_list + control_list

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not data:
  # agent parameters
  agent_kw = 'basic'  #@param ['basic', 'quad_q'] 
  gen_alpha = .25 #@param
  gen_beta = 3 #@param
  forget_rate = 0. #@param
  perseverance_bias = 0.25 #@param
  # environment parameters
  non_binary_reward = False #@param
  n_actions = 2 #@param
  sigma = .1  #@param

  # dataset parameters
  n_trials_per_session = 200  #@param
  n_sessions = 256  #@param

  # setup
  environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_rewards=non_binary_reward)
  agent = bandits.AgentQ(gen_alpha, gen_beta, n_actions, forget_rate, perseverance_bias)  

  dataset_train, experiment_list_train = bandits.create_dataset(
      agent=agent,
      environment=environment,
      n_trials_per_session=n_trials_per_session,
      n_sessions=n_sessions,
      sequence_length=n_steps_per_call,
      device=device)

  dataset_test, experiment_list_test = bandits.create_dataset(
      agent=agent,
      environment=environment,
      n_trials_per_session=200,
      n_sessions=1024,
      sequence_length=n_trials_per_session,
      device=device)
  
  params_path = rnn_utils.parameter_file_naming(
      'params/params',
      use_lstm,
      last_output,
      last_state,
      gen_beta,
      forget_rate,
      perseverance_bias,
      non_binary_reward,
      verbose=True,
  )
  
else:
  # load data
  with open(path_data, 'rb') as f:
      dataset_train = pickle.load(f)

# define model
if use_lstm:
  model = rnn.LSTM(
      n_actions=n_actions, 
      hidden_size=hidden_size, 
      init_value=0.5,
      device=device,
      ).to(device)
else:
  model = [rnn.RLRNN(
      n_actions=n_actions, 
      hidden_size=hidden_size, 
      init_value=0.5,
      last_output=last_output,
      last_state=last_state,
      device=device,
      list_sindy_signals=sindy_feature_list,
      ).to(device)
           for _ in range(n_submodels)]

optimizer_rnn = None#torch.optim.Adam(model.parameters(), lr=learning_rate)

if train:
  if checkpoint:
    # load trained parameters
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    optimizer_rnn.load_state_dict(state_dict['optimizer'])
    print('Loaded parameters.')
  
  start_time = time.time()
  
  #Fit the hybrid RNN
  print('Training the hybrid RNN...')
  model, optimizer_rnn, _ = rnn_training.fit_model(
      model=model,
      dataset=dataset_train,
      optimizer=optimizer_rnn,
      convergence_threshold=convergence_threshold,
      epochs=epochs,
      batch_size=batch_size,
      n_submodels=n_submodels,
      return_ensemble=ensemble,
      voting_type=voting_type,
      sampling_replacement=sampling_replacement,
  )
  
  # validate model
  print('\nValidating the trained hybrid RNN on a test dataset...')
  with torch.no_grad():
    rnn_training.fit_model(
        model=model,
        dataset=dataset_test,
    )

  # print adjusted beta parameter
  # if isinstance(model, rnn.RLRNN):
  #   print(f'beta: {np.round(model.beta.item(), 2)}')
  # elif isinstance(model, rnn.EnsembleRNN):
  #   beta = torch.tensor([model_i.beta for model_i in model]).reshape(1, -1)
  #   print(f'beta: {np.round(model.vote(beta, voting_type).item(), 2)}')
  
  print(f'Training took {time.time() - start_time:.2f} seconds.')
  
  # save trained parameters  
  state_dict = {
    'model': model.state_dict() if isinstance(model, torch.nn.Module) else [model_i.state_dict() for model_i in model],
    'optimizer': optimizer_rnn.state_dict() if isinstance(optimizer_rnn, torch.optim.Adam) else [optim_i.state_dict() for optim_i in optimizer_rnn],
  }
  torch.save(state_dict, params_path)
  
  print(f'Saved RNN parameters to file {params_path}.')

else:
  # load trained parameters
  model.load_state_dict(torch.load(params_path)['model'])
  print(f'Loaded parameters from file {params_path}.')

# if hasattr(model, 'beta'):
#   print(f'beta: {model.beta}')

# Synthesize a dataset using the fitted network
environment = bandits.EnvironmentBanditsDrift(0.1)
model.set_device(torch.device('cpu'))
model.to(torch.device('cpu'))
rnn_agent = bandits.AgentNetwork(model, n_actions=2)
# dataset_rnn, experiment_list_rnn = bandits.create_dataset(rnn_agent, environment, 220, 10)

# Analysis
session_id = 0

choices = experiment_list_test[session_id].choices
rewards = experiment_list_test[session_id].rewards

list_probs = []
list_qs = []

# get q-values from groundtruth
qs_test, probs_test = bandits.get_update_dynamics(experiment_list_test[session_id], agent)
list_probs.append(np.expand_dims(probs_test, 0))
list_qs.append(np.expand_dims(qs_test, 0))

# get q-values from trained rnn
qs_rnn, probs_rnn = bandits.get_update_dynamics(experiment_list_test[session_id], rnn_agent)
list_probs.append(np.expand_dims(probs_rnn, 0))
list_qs.append(np.expand_dims(qs_rnn, 0))

colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

# concatenate all choice probs and q-values
probs = np.concatenate(list_probs, axis=0)
qs = np.concatenate(list_qs, axis=0)

# normalize q-values
# qs = (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

fig, axs = plt.subplots(4, 1, figsize=(20, 10))

reward_probs = np.stack([experiment_list_test[session_id].timeseries[:, i] for i in range(n_actions)], axis=0)
bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=reward_probs,
    timeseries_name='Reward Probs',
    labels=[f'Arm {a}' for a in range(n_actions)],
    color=['tab:purple', 'tab:cyan'],
    binary=not non_binary_reward,
    fig_ax=(fig, axs[0]),
    )

bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=probs[:, :, 0],
    timeseries_name='Choice Probs',
    color=colors,
    labels=['Ground Truth', 'RNN'],
    binary=not non_binary_reward,
    fig_ax=(fig, axs[1]),
    )

bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=qs[:, :, 0],
    timeseries_name='Q-Values',
    color=colors,
    binary=not non_binary_reward,
    fig_ax=(fig, axs[2]),
    )

dqs_arms = -1*np.diff(qs, axis=2)

bandits.plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=dqs_arms[:, :, 0],
    timeseries_name='dQ/dActions',
    color=colors,
    binary=not non_binary_reward,
    fig_ax=(fig, axs[3]),
    )

plt.show()
