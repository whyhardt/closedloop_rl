#@title Import libraries
import sys
import os
import warnings

import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

# warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import rnn, rnn_training, bandits, rnn_utils

def main(
  train = True,
  checkpoint = False,

  # rnn parameters
  hidden_size = 4,
  dropout = 0.25,
  use_lstm = False,

  # ensemble parameters
  evolution_interval = 100,
  sampling_replacement = False,
  n_submodels = 1,
  init_population = -1,
  ensemble = rnn_training.ensembleTypes.BEST,  # Options; .BEST (picking best submodel after training), .AVERAGE (averaging the parameters of all submodels after each epoch), .VOTE (keeping all models but voting at each time step after being trained)
  voting_type = rnn.EnsembleRNN.MEDIAN,  # Options: .MEAN, .MEDIAN; applies only for ensemble==rnn_training.ensemble_types.VOTE

  # training parameters
  # dataset parameters (if None, they will be created)
  dataset_train = None,
  dataset_val = None,
  dataset_test = None,
  experiment_list_test = None,
  n_trials_per_session = 50,
  n_sessions = 4096,
  epochs = 1024,
  n_steps_per_call = 8,  # None for full sequence
  batch_size = None,  # None for one batch per epoch
  learning_rate = 1e-4,
  convergence_threshold = 1e-6,
  
  # ground truth parameters
  alpha = 0.25,
  beta = 3,
  forget_rate = 0.1,
  perseveration_bias = 0.25,
  correlated_update = False,
  regret = True,
  
  # environment parameters
  n_actions = 2,
  sigma = 0.2,
  correlated_reward = False,
  non_binary_reward = False,

  analysis: bool = False,
  ):

  if not os.path.exists('params'):
    os.makedirs('params')

  # tracked variables in the RNN
  x_train_list = ['xQf','xQr', 'xQr_r', 'xQr_p', 'xH']
  control_list = ['ca', 'cr', 'cdQr[k-1]', 'cdQr[k-2]']
  sindy_feature_list = x_train_list + control_list

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if init_population == -1:
    init_population = n_submodels
  elif init_population < n_submodels:
    raise ValueError(f'init_population ({init_population}) must be greater or equal to n_submodels ({n_submodels}).')

  # setup
  environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_reward=non_binary_reward, correlated_reward=correlated_reward)
  agent = bandits.AgentQ(alpha, beta, n_actions, forget_rate, perseveration_bias, correlated_update, regret)  
  print('Setup of the environment and agent complete.')

  if dataset_train is None:    
    print('Creating the training dataset...', end='\r')
    dataset_train, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions,
        device=device)    

  if dataset_val is None:
    print('Creating the validation dataset...', end='\r')
    dataset_val, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=50,
        n_sessions=16,
        device=device)
    
  if dataset_test is None:
    print('Creating the test dataset...', end='\r')
    dataset_test, experiment_list_test = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=200,
        n_sessions=128,
        device=device)
  
  print('Setup of datasets complete.')
  
  params_path = rnn_utils.parameter_file_naming(
      'params/params',
      use_lstm,
      alpha,
      beta,
      forget_rate,
      perseveration_bias,
      correlated_update,
      regret,
      non_binary_reward,
      verbose=True,
  )

  if ensemble > -1 and n_submodels == 1:
    Warning('Ensemble is actived but n_submodels is set to 1. Deactivated ensemble.')
    ensemble = rnn_training.ensembleTypes.BEST

  # define model
  if use_lstm:
    model = rnn.LSTM(
        n_actions=n_actions, 
        hidden_size=hidden_size, 
        init_value=0.,
        device=device,
        ).to(device)
  else:
    model = [rnn.RLRNN(
        n_actions=n_actions, 
        hidden_size=hidden_size, 
        init_value=0.5,
        device=device,
        list_sindy_signals=sindy_feature_list,
        dropout=dropout,
        ).to(device)
            for _ in range(init_population)]

  optimizer_rnn = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in model]

  print('Setup of the RNN model complete.')

  if checkpoint:
      model, optimizer_rnn = rnn_utils.load_checkpoint(params_path, model, optimizer_rnn, voting_type)
      print('Loaded model parameters.')

  if train:
    
    start_time = time.time()
    
    #Fit the hybrid RNN
    print('Training the hybrid RNN...')
    for m in model:
      m.train()
    model, optimizer_rnn, _ = rnn_training.fit_model(
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_val,
        optimizer=optimizer_rnn,
        convergence_threshold=convergence_threshold,
        epochs=epochs,
        batch_size=batch_size,
        n_submodels=n_submodels,
        ensemble_type=ensemble,
        voting_type=voting_type,
        sampling_replacement=sampling_replacement,
        evolution_interval=evolution_interval,
        n_steps_per_call=n_steps_per_call,
    )
    
    # save trained parameters  
    state_dict = {
      'model': model.state_dict() if isinstance(model, torch.nn.Module) else [model_i.state_dict() for model_i in model],
      'optimizer': optimizer_rnn.state_dict() if isinstance(optimizer_rnn, torch.optim.Adam) else [optim_i.state_dict() for optim_i in optimizer_rnn],
    }
    torch.save(state_dict, params_path)
    
    print(f'Saved RNN parameters to file {params_path}.')
    
    # validate model
    print('\nTesting the trained hybrid RNN on a test dataset...')
    if isinstance(model, list):
      for m in model:
        m.eval()
    else:
      model.eval()
    with torch.no_grad():
      _, _, loss_test = rnn_training.fit_model(
          model=model,
          dataset_train=dataset_test,
          dataset_test=None,
          n_steps_per_call=1,
      )

    print(f'Training took {time.time() - start_time:.2f} seconds.')
  # else:
  #   model, _, _ = rnn_training.fit_model(
  #       model=model,
  #       dataset_train=dataset_train,
  #       dataset_test=None,
  #       epochs=0,
  #       n_submodels=n_submodels,
  #       ensemble_type=ensemble,
  #       voting_type=voting_type,
  #       verbose=True
  #   )

  if analysis:
    # Synthesize a dataset using the fitted network
    environment = bandits.EnvironmentBanditsDrift(sigma=sigma)
    model.set_device(torch.device('cpu'))
    model.to(torch.device('cpu'))
    rnn_agent = bandits.AgentNetwork(model, n_actions=n_actions, deterministic=True)

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
    def normalize(qs):
      return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

    # qs = normalize(qs)

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
        timeseries_name='Q Arm 0',
        color=colors,
        binary=not non_binary_reward,
        fig_ax=(fig, axs[2]),
        )

    # bandits.plot_session(
    #     compare=True,
    #     choices=choices,
    #     rewards=rewards,
    #     timeseries=qs[:, :, 1],
    #     timeseries_name='Q Arm 1',
    #     color=colors,
    #     binary=not non_binary_reward,
    #     fig_ax=(fig, axs[3]),
    #     )

    dqs_t = np.diff(qs, axis=1)

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=dqs_t[:, :, 0],
        timeseries_name='dQ/dt',
        color=colors,
        binary=not non_binary_reward,
        fig_ax=(fig, axs[3]),
        )

    # dqs_arms = normalize(-1*np.diff(qs, axis=2))

    # bandits.plot_session(
    #     compare=True,
    #     choices=choices,
    #     rewards=rewards,
    #     timeseries=dqs_arms[:, :, 0],
    #     timeseries_name='dQ/dActions',
    #     color=colors,
    #     binary=not non_binary_reward,
    #     fig_ax=(fig, axs[3]),
    #     )

    plt.show()

  return loss_test


if __name__=='__main__':
  main(
    checkpoint=False,
    n_submodels=4,
    epochs=1024,
    sampling_replacement=True,
    ensemble=rnn_training.ensembleTypes.AVERAGE,
    evolution_interval=None,
    analysis=True,
  )