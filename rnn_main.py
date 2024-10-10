#@title Import libraries
import sys
import os
import warnings

import pickle
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Collection, Callable
import argparse

# warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import rnn, rnn_training, bandits, rnn_utils

def main(
  checkpoint = False,
  model = None,

  # rnn parameters
  hidden_size = 4,
  dropout = 0.25,
  use_lstm = False,

  # ensemble parameters
  evolution_interval = 100,
  n_submodels = 1,
  init_population = -1,
  ensemble = rnn_training.ensembleTypes.BEST,  # Options; .BEST (picking best submodel after training), .AVERAGE (averaging the parameters of all submodels after each epoch), .VOTE (keeping all models but voting at each time step after being trained)
  voting_type = rnn.EnsembleRNN.MEDIAN,  # Options: .MEAN, .MEDIAN; applies only for ensemble==rnn_training.ensemble_types.VOTE

  # training parameters
  dataset_train = None,
  dataset_val = None,
  dataset_test = None,
  experiment_list_test = None,
  
  n_trials_per_session = 64,
  n_sessions = 128,
  bagging = False,
  n_oversampling = -1,
  epochs = 1024,
  n_steps_per_call = -1,  # -1 for full sequence
  batch_size = -1,  # -1 for one batch per epoch
  learning_rate = 1e-3,
  convergence_threshold = 1e-6,
  
  # ground truth parameters
  alpha = 0.25,
  beta = 3,
  forget_rate = 0.,
  perseveration_bias = 0.,
  regret = False,
  confirmation_bias = False,
  reward_update_rule: Callable = None,
  
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
  x_train_list = ['xQf', 'xLR', 'xH', 'xHa', 'xHn']
  control_list = ['ca', 'cr', 'cQ', 'cp', 'ca_prev']
  for i in range(hidden_size):
    x_train_list.append(f'xLR_{i}')
  sindy_feature_list = x_train_list + control_list

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if init_population == -1:
    init_population = n_submodels
  elif init_population < n_submodels:
    raise ValueError(f'init_population ({init_population}) must be greater or equal to n_submodels ({n_submodels}).')

  # setup
  environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_reward=non_binary_reward, correlated_reward=correlated_reward)
  # environment = bandits.EnvironmentBanditsSwitch(0.1)
  agent = bandits.AgentQ(n_actions, alpha, beta, forget_rate, perseveration_bias, regret, confirmation_bias)  
  if reward_update_rule is not None:
    agent.set_reward_update(reward_update_rule)
  print('Setup of the environment and agent complete.')

  if epochs > 0:
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
          n_trials_per_session=64,
          n_sessions=16,
          device=device)
    
  if dataset_test is None:
    print('Creating the test dataset...', end='\r')
    dataset_test, experiment_list_test = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=200,
        n_sessions=1024,
        device=device)
  
  print('Setup of datasets complete.')
  
  if model is None:
    params_path = rnn_utils.parameter_file_naming(
        'params/params',
        use_lstm,
        alpha,
        beta,
        forget_rate,
        perseveration_bias,
        regret,
        confirmation_bias,
        non_binary_reward,
        verbose=True,
    )
  else:
    params_path = model

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

  loss_test = None
  if epochs > 0:
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
        bagging=bagging,
        n_oversampling=n_oversampling,
        evolution_interval=evolution_interval,
        n_steps_per_call=n_steps_per_call,
    )
    
    # save trained parameters  
    state_dict = {
      'model': model.state_dict() if isinstance(model, torch.nn.Module) else [model_i.state_dict() for model_i in model],
      'optimizer': optimizer_rnn.state_dict() if isinstance(optimizer_rnn, torch.optim.Adam) else [optim_i.state_dict() for optim_i in optimizer_rnn],
      # 'groundtruth': {
      #   'alpha': alpha,
      #   'beta': beta,
      #   'forget_rate': forget_rate,
      #   'perseveration_bias': perseveration_bias,
      #   'regret': regret,
      #   'confirmation_bias': confirmation_bias,
      #   'reward_update_rule': agent._reward_update,
      # },
    }
    torch.save(state_dict, params_path)
    
    print(f'Saved RNN parameters to file {params_path}.')

    print(f'Training took {time.time() - start_time:.2f} seconds.')
  else:
    if isinstance(model, list):
      model = model[0]
      optimizer_rnn = optimizer_rnn[0]
  
  print(f'Trained beta of RNN is: {model.beta.item()}')
  
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
  
  # -----------------------------------------------------------
  # Analysis
  # -----------------------------------------------------------
  
  if analysis:
    # Synthesize a dataset using the fitted network
    environment = bandits.EnvironmentBanditsDrift(sigma=sigma)
    model.set_device(torch.device('cpu'))
    model.to(torch.device('cpu'))
    rnn_agent = bandits.AgentNetwork(model, n_actions=n_actions, deterministic=True)
    
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

    reward_probs = np.stack([experiment_list_test[session_id].reward_probabilities[:, i] for i in range(n_actions)], axis=0)
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
    
    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=qs[:, :, 0]-qs[:, :, 1],
        timeseries_name='dQ/dArm',
        color=colors,
        binary=not non_binary_reward,
        fig_ax=(fig, axs[3]),
        )
    
    # dqs_t = np.diff(qs, axis=1)

    # bandits.plot_session(
    #     compare=True,
    #     choices=choices,
    #     rewards=rewards,
    #     timeseries=dqs_t[:, :, 0],
    #     timeseries_name='dQ/dt',
    #     color=colors,
    #     binary=not non_binary_reward,
    #     fig_ax=(fig, axs[3]),
    #     )

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
  
  parser = argparse.ArgumentParser(description='Trains the RNN on behavioral data to uncover the underlying Q-Values via different cognitive mechanisms.')
  
  # Training parameters
  parser.add_argument('--checkpoint', action='store_true', help='Whether to load a checkpoint')
  parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
  parser.add_argument('--epochs', type=int, default=128, help='Number of epochs for training')
  parser.add_argument('--n_trials_per_session', type=int, default=64, help='Number of trials per session')
  parser.add_argument('--n_sessions', type=int, default=4096, help='Number of sessions')
  parser.add_argument('--n_steps_per_call', type=int, default=8, help='Number of steps per call')
  parser.add_argument('--bagging', action='store_true', help='Whether to use bagging')
  parser.add_argument('--n_oversampling', type=int, default=-1, help='Number of oversampling iterations')
  parser.add_argument('--batch_size', type=int, default=-1, help='Batch size')
  parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate of the RNN')

  # Ensemble parameters
  parser.add_argument('--n_submodels', type=int, default=8, help='Number of submodels in the ensemble')
  parser.add_argument('--ensemble', type=int, default=1, help='Defines the type of ensembling. Options -- -1: take the best model; 0: let the submodels vote (median); 1: average the parameters after each epoch (recommended)')

  # RNN parameters
  parser.add_argument('--hidden_size', type=int, default=8, help='Hidden size of the RNN')
  parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

  # Ground truth parameters
  parser.add_argument('--alpha', type=float, default=0.25, help='Alpha parameter for the Q-learning update rule')
  parser.add_argument('--beta', type=float, default=3, help='Beta parameter for the Q-learning update rule')
  parser.add_argument('--forget_rate', type=float, default=0., help='Forget rate')
  parser.add_argument('--perseveration_bias', type=float, default=0., help='Perseveration bias')
  parser.add_argument('--regret', action='store_true', help='Whether to include regret')
  parser.add_argument('--confirmation_bias', action='store_true', help='Whether to include confirmation bias')

  # Environment parameters
  parser.add_argument('--sigma', type=float, default=0.1, help='Drift rate of the reward probabilities')
  parser.add_argument('--non_binary_reward', action='store_true', help='Whether to use non-binary rewards')

  # Analysis parameters
  parser.add_argument('--analysis', action='store_true', help='Whether to perform analysis')

  args = parser.parse_args()  
  
  main(
    # train = True, 
    checkpoint = args.checkpoint,
    # model = 'params/params_rnn_a025_b3_cb.pkl',
    model = args.model,
    
    # training parameters
    epochs=args.epochs,
    n_trials_per_session = args.n_trials_per_session,
    n_sessions = args.n_sessions,
    n_steps_per_call = args.n_steps_per_call,
    bagging = args.bagging,
    n_oversampling=args.n_oversampling,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,

    # ensemble parameters
    n_submodels=args.n_submodels,
    ensemble=args.ensemble,
    
    # rnn parameters
    hidden_size = args.hidden_size,
    dropout = args.dropout,
    
    # ground truth parameters
    alpha = args.alpha,
    beta = args.beta,
    forget_rate = args.forget_rate,
    perseveration_bias = args.perseveration_bias,
    regret = args.regret,
    confirmation_bias = args.confirmation_bias,
    # reward_update_rule = lambda q, reward: reward-q,
    
    # environment parameters
    sigma = args.sigma,
    non_binary_reward = args.non_binary_reward,
    
    analysis = args.analysis,
  )
  