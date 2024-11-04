#@title Import libraries
import sys
import os

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import argparse

# warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import rnn, rnn_training, bandits, rnn_utils
from utils import convert_dataset

def main(
  checkpoint = False,
  model = None,
  data = None,

  # rnn parameters
  hidden_size = 8,
  dropout = 0.1,
  use_lstm = False,

  # ensemble parameters
  evolution_interval = -1,
  n_submodels = 1,
  init_population = -1,
  ensemble = rnn_training.ensembleTypes.AVERAGE,  # Options; .BEST (picking best submodel after training), .AVERAGE (averaging the parameters of all submodels after each epoch), .VOTE (keeping all models but voting at each time step after being trained)
  voting_type = rnn.EnsembleRNN.MEDIAN,  # Options: .MEAN, .MEDIAN; applies only for ensemble==rnn_training.ensemble_types.VOTE

  # training parameters
  dataset_train = None,
  dataset_val = None,
  dataset_test = None,
  experiment_list_test = None,
  
  # data and training parameters
  epochs = 128,
  n_trials_per_session = 64,
  n_sessions = 4096,
  bagging = False,
  n_oversampling = -1,
  n_steps_per_call = 16,  # -1 for full sequence
  batch_size = -1,  # -1 for one batch per epoch
  learning_rate = 1e-2,
  convergence_threshold = 1e-6,
  weight_decay = 0.,
  
  # ground truth parameters
  alpha = 0.25,
  beta = 3.,
  forget_rate = 0.,
  perseveration_bias = 0.,
  alpha_penalty = -1.,
  confirmation_bias = 0.,
  reward_prediction_error: Callable = None,
  
  # environment parameters
  n_actions = 2,
  sigma = 0.1,
  correlated_reward = False,
  non_binary_reward = False,

  analysis: bool = False,
  ):

  if not os.path.exists('params'):
    os.makedirs('params')
  
  # tracked variables in the RNN
  x_train_list = ['xQf', 'xLR', 'xH', 'xHf', 'xU', 'xUf', 'xB']
  control_list = ['ca', 'cr', 'cp', 'ca_repeat', 'cQ', 'cU_0', 'cU_1']
  # for i in range(hidden_size):
  #   x_train_list.append(f'xLR_{i}')
  sindy_feature_list = x_train_list + control_list

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if init_population == -1:
    init_population = n_submodels
  elif init_population < n_submodels:
    raise ValueError(f'init_population ({init_population}) must be greater or equal to n_submodels ({n_submodels}).')
  
  if data is None:
    # setup
    environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_reward=non_binary_reward, correlated_reward=correlated_reward)
    # environment = bandits.EnvironmentBanditsSwitch(sigma)
    agent = bandits.AgentQ(n_actions, alpha, beta, forget_rate, perseveration_bias, alpha_penalty, confirmation_bias)  
    if reward_prediction_error is not None:
      agent.set_reward_prediction_error(reward_prediction_error)
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
  else:
    dataset, experiment_list_test = convert_dataset.to_datasetrnn(data)
    indexes_dataset = np.arange(len(dataset.xs))
    np.random.shuffle(indexes_dataset)
    
    xs_train, ys_train = dataset.xs[indexes_dataset[:int(0.95*len(dataset.xs))]], dataset.ys[indexes_dataset[:int(0.95*len(dataset.xs))]]
    xs_val, ys_val = dataset.xs[indexes_dataset[int(0.95*len(dataset.xs)):]], dataset.ys[indexes_dataset[int(0.95*len(dataset.xs)):]]
    xs_test, ys_test = xs_val, ys_val
    dataset_train = rnn_utils.DatasetRNN(xs_train, ys_train)
    dataset_val = rnn_utils.DatasetRNN(xs_val, ys_val)
    dataset_test = rnn_utils.DatasetRNN(xs_test, ys_test)
    
    # xs_train, ys_train = dataset.xs[indexes_dataset[:int(0.8*len(dataset.xs))]], dataset.ys[indexes_dataset[:int(0.8*len(dataset.xs))]]
    # xs_val, ys_val = dataset.xs[indexes_dataset[int(0.8*len(dataset.xs)):int(0.9*len(dataset.xs))]], dataset.ys[indexes_dataset[int(0.8*len(dataset.xs)):int(0.9*len(dataset.xs))]]
    # xs_test, ys_test = dataset.xs[indexes_dataset[int(0.9*len(dataset.xs)):]], dataset.ys[indexes_dataset[int(0.9*len(dataset.xs)):]]
    # dataset_train = rnn_utils.DatasetRNN(xs_train, ys_train)
    # dataset_val = rnn_utils.DatasetRNN(xs_val, ys_val)
    # dataset_test = rnn_utils.DatasetRNN(xs_test, ys_test)
    
  if model is None:
    params_path = rnn_utils.parameter_file_naming(
        'params/params',
        use_lstm,
        alpha,
        beta,
        forget_rate,
        perseveration_bias,
        alpha_penalty,
        confirmation_bias,
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

  optimizer_rnn = [torch.optim.Adam(m.parameters(), lr=learning_rate, weight_decay=weight_decay) for m in model]

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
  
  print(f'Trained initial beta of RNN is: {model.beta.item()}')
  
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
    
    labels = ['Ground Truth', 'RNN'] if data is None else ['RNN'] 
    session_id = 0

    choices = experiment_list_test[session_id].choices
    rewards = experiment_list_test[session_id].rewards

    list_probs = []
    list_Qs = []
    list_qs = []
    list_hs = []
    list_us = []
    list_bs = []

    # get q-values from groundtruth
    if data is None:
      qs_test, probs_test, _ = bandits.get_update_dynamics(experiment_list_test[session_id], agent)
      list_probs.append(np.expand_dims(probs_test, 0))
      list_Qs.append(np.expand_dims(qs_test[0], 0))
      list_qs.append(np.expand_dims(qs_test[1], 0))
      list_hs.append(np.expand_dims(qs_test[2], 0))
      list_us.append(np.expand_dims(qs_test[3], 0))
      list_bs.append(np.expand_dims(qs_test[4], 0))

    # get q-values from trained rnn
    qs_rnn, probs_rnn, _ = bandits.get_update_dynamics(experiment_list_test[session_id], rnn_agent)
    list_probs.append(np.expand_dims(probs_rnn, 0))
    list_Qs.append(np.expand_dims(qs_rnn[0], 0))
    list_qs.append(np.expand_dims(qs_rnn[1], 0))
    list_hs.append(np.expand_dims(qs_rnn[2], 0))
    list_us.append(np.expand_dims(qs_rnn[3], 0))
    list_bs.append(np.expand_dims(qs_rnn[4], 0))

    # concatenate all choice probs and q-values
    probs = np.concatenate(list_probs, axis=0)
    Qs = np.concatenate(list_Qs, axis=0)
    qs = np.concatenate(list_qs, axis=0)
    hs = np.concatenate(list_hs, axis=0)
    us = np.concatenate(list_us, axis=0)
    bs = np.concatenate(list_bs, axis=0)

    colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']
    
    # normalize q-values
    def normalize(qs):
      return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

    # qs = normalize(qs)

    fig, axs = plt.subplots(7, 1, figsize=(20, 10))

    reward_probs = np.stack([experiment_list_test[session_id].reward_probabilities[:, i] for i in range(n_actions)], axis=0)
    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=reward_probs,
        timeseries_name='p(R)',
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
        timeseries_name='p(A)',
        color=colors,
        labels=labels,
        binary=not non_binary_reward,
        fig_ax=(fig, axs[1]),
        )

    bandits.plot_session(
      compare=True,
      choices=choices,
      rewards=rewards,
      timeseries=Qs[:, :, 0],
      timeseries_name='Q',
      color=colors,
      binary=True,
      fig_ax=(fig, axs[2]),
      )

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=qs[:, :, 0],
        timeseries_name='q',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[3]),
        )

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=hs[:, :, 0],
        timeseries_name='a',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[4]),
        )

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=us[:, :, 0],
        timeseries_name='u',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[5]),
    )
    
    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=bs[:, :, 0],
        timeseries_name='b',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[6]),
    )

    plt.show()
    
    if data is None:
      print(f'Average beta of ground truth is: beta_avg = {np.round(np.mean(qs_test[4]), 2)}')
    print(f'Average beta of RNN is: beta_avg = {np.round(np.mean(qs_rnn[4]), 2)}')
    
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
  parser.add_argument('--alpha_p', type=float, default=-1., help='Learning rate for negative outcomes; if -1: same as alpha')
  parser.add_argument('--confirmation_bias', type=float, default=0., help='Whether to include confirmation bias')

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
    alpha_penalty = args.regret,
    confirmation_bias = args.confirmation_bias,
    # reward_update_rule = lambda q, reward: reward-q,
    
    # environment parameters
    sigma = args.sigma,
    non_binary_reward = args.non_binary_reward,
    
    analysis = args.analysis,
  )
  