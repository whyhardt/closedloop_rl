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
  epochs_train = 128,
  epochs_finetune = 0,
  n_trials_per_session = 64,
  n_sessions = 4096,
  bagging = False,
  n_oversampling_train = -1,
  n_oversampling_finetune = -1,
  n_steps_per_call = 16,  # -1 for full sequence
  batch_size_train = -1,  # -1 for one batch per epoch
  batch_size_finetune = -1,
  lr_train = 1e-2,
  lr_finetune = 1e-4,
  convergence_threshold = 1e-6,
  weight_decay = 0.,
  parameter_variance = 0.,
  
  # ground truth parameters
  alpha = 0.25,
  beta = 3.,
  forget_rate = 0.,
  perseverance_bias = 0.,
  alpha_penalty = -1.,
  confirmation_bias = 0.,
  reward_prediction_error: Callable = None,
  
  # environment parameters
  n_actions = 2,
  sigma = 0.1,

  analysis: bool = False,
  ):

  if not os.path.exists('params'):
    os.makedirs('params')
  
  # tracked variables in the RNN
  x_train_list = ['xQf', 'xLR', 'xC', 'xCf']
  control_list = ['ca', 'cr', 'cp', 'ca_repeat', 'cQ']
  sindy_feature_list = x_train_list + control_list

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if init_population == -1:
    init_population = n_submodels
  elif init_population < n_submodels:
    raise ValueError(f'init_population ({init_population}) must be greater or equal to n_submodels ({n_submodels}).')
  
  if data is None:
    # setup
    environment = bandits.EnvironmentBanditsDrift(sigma, n_actions)
    # environment = bandits.EnvironmentBanditsSwitch(sigma)
    agent = bandits.AgentQ(n_actions, alpha, beta, forget_rate, perseverance_bias, alpha_penalty, confirmation_bias, parameter_variance)  
    agent_finetune = bandits.AgentQ(n_actions, alpha, beta, forget_rate, perseverance_bias, alpha_penalty, confirmation_bias)  
    if reward_prediction_error is not None:
      agent.set_reward_prediction_error(reward_prediction_error)
    print('Setup of the environment and agent complete.')

    if epochs_train > 0:
      if dataset_train is None:    
        print('Creating the training dataset...')
        dataset_train, experiment_list_train, parameter_list_train = bandits.create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions,
            sample_parameters=parameter_variance!=0,
            device=device)

      if dataset_val is None:
        print('Creating the validation dataset...')
        dataset_val, _, parameter_list_val = bandits.create_dataset(
            agent=agent_finetune,
            environment=environment,
            n_trials_per_session=64,
            n_sessions=16,
            device=device)
      
    if dataset_test is None:
      print('Creating the test dataset...')
      dataset_test, experiment_list_test, parameter_list_test = bandits.create_dataset(
          agent=agent_finetune,
          environment=environment,
          n_trials_per_session=200,
          n_sessions=1024,
          device=device)
      
    print('Creating the finetune dataset...')
    dataset_fine, experiment_list_fine, parameter_list_fine = bandits.create_dataset(
          agent=agent_finetune,
          environment=environment,
          n_trials_per_session=n_trials_per_session,
          n_sessions=1,
          device=device)
    
    print('Setup of datasets complete.')
  else:
    dataset, experiment_list_test, df, update_dynamics = convert_dataset.convert_dataset(data)
    indexes_dataset = np.arange(len(dataset.xs))
    # np.random.shuffle(indexes_dataset)
    
    # idx_train = int(0.95*len(dataset.xs))
    idx_train = -1
    experiment_list_test = experiment_list_test[idx_train:]
    xs_train, ys_train = dataset.xs[indexes_dataset[:idx_train]], dataset.ys[indexes_dataset[:idx_train]]
    xs_val, ys_val = dataset.xs[indexes_dataset[idx_train:]], dataset.ys[indexes_dataset[idx_train:]]
    xs_test, ys_test = xs_val, ys_val
    dataset_train = rnn_utils.DatasetRNN(xs_train, ys_train)
    dataset_val = rnn_utils.DatasetRNN(xs_val, ys_val, sequence_length=64)
    dataset_test = rnn_utils.DatasetRNN(xs_test, ys_test)
    
    # check if groundtruth parameters in data - only applicable to generated data with e.g. utils/create_dataset.py
    if 'mean_beta' in df.columns:
      beta = df['beta'].values[idx_train]
      alpha = df['alpha'].values[idx_train]
      alpha_penalty = df['alpha_penalty'].values[idx_train]
      confirmation_bias = df['confirmation_bias'].values[idx_train]
      forget_rate = df['forget_rate'].values[idx_train]
      perseverance_bias = df['perseverance_bias'].values[idx_train]
    
  if model is None:
    params_path = rnn_utils.parameter_file_naming('params/params',alpha,beta,forget_rate,perseverance_bias,alpha_penalty,confirmation_bias,parameter_variance,verbose=True)
  else:
    params_path = model

  if ensemble > -1 and n_submodels == 1:
    Warning('Ensemble is actived but n_submodels is set to 1. Deactivated ensemble.')
    ensemble = rnn_training.ensembleTypes.BEST

  # define model
  model = [rnn.RLRNN(
      n_actions=n_actions, 
      hidden_size=hidden_size, 
      init_value=0.5,
      device=device,
      list_sindy_signals=sindy_feature_list,
      dropout=dropout,
      ).to(device)
          for _ in range(init_population)]

  optimizer_rnn = [torch.optim.Adam(m.parameters(), lr=lr_train, weight_decay=weight_decay) for m in model]

  print('Setup of the RNN model complete.')

  if checkpoint:
      model, optimizer_rnn = rnn_utils.load_checkpoint(params_path, model, optimizer_rnn, voting_type)
      print('Loaded model parameters.')

  loss_test = None
  if epochs_train > 0:
    start_time = time.time()
    
    #Fit the RNN
    print('Training the RNN...')
    for m in model:
      m.train()
    model, optimizer_rnn, _ = rnn_training.fit_model(
        model=model,
        dataset_train=dataset_train,
        # dataset_test=dataset_test,
        optimizer=optimizer_rnn,
        convergence_threshold=convergence_threshold,
        epochs=epochs_train,
        batch_size=batch_size_train,
        n_submodels=n_submodels,
        ensemble_type=ensemble,
        voting_type=voting_type,
        bagging=bagging,
        n_oversampling=n_oversampling_train,
        evolution_interval=evolution_interval,
        n_steps_per_call=n_steps_per_call,
    )
    
    # save trained parameters
    state_dict = {
      'model': model.state_dict() if isinstance(model, torch.nn.Module) else [model_i.state_dict() for model_i in model],
      'optimizer': optimizer_rnn.state_dict() if isinstance(optimizer_rnn, torch.optim.Adam) else [optim_i.state_dict() for optim_i in optimizer_rnn],
      'groundtruth': {
        'alpha': alpha,
        'beta': beta,
        'forget_rate': forget_rate,
        'perseverance_bias': perseverance_bias,
        'alpha_penalty': alpha_penalty,
        'confirmation_bias': confirmation_bias,
      },
    }
    print('Training finished.')
    print(f'Trained beta of RNN is: {model._beta_reward.item()} and {model._beta_choice.item()}')
    torch.save(state_dict, params_path)
    print(f'Saved RNN parameters to file {params_path}.')
    print(f'Training took {time.time() - start_time:.2f} seconds.')
  else:
    if isinstance(model, list):
      model = model[0]
      optimizer_rnn = optimizer_rnn[0]
  
  # validate model
  print('\nTesting the trained RNN on the test dataset...')
  model.eval()
  with torch.no_grad():
    _, _, loss_test = rnn_training.fit_model(
        model=model,
        dataset_train=dataset_test,
    )
  
  # Finetune the RNN with individual data
  if epochs_finetune > 0:
    start_time = time.time()
    print('Finetuning the RNN...')
    for subnetwork in x_train_list:
      model.finetune_training(subnetwork, keep_dropout=True)
    optimizer_fine = [torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_finetune, weight_decay=weight_decay)]
    model, optimizer_fine, _ = rnn_training.fit_model(
        model=[model],
        dataset_train=dataset_val,
        # dataset_test=dataset_test,
        optimizer=optimizer_fine,
        convergence_threshold=convergence_threshold,
        epochs=epochs_finetune,
        batch_size=batch_size_finetune,
        n_submodels=n_submodels,
        ensemble_type=ensemble,
        voting_type=voting_type,
        bagging=bagging,
        n_oversampling=n_oversampling_finetune,
        evolution_interval=evolution_interval,
        n_steps_per_call=n_steps_per_call,
    )
    model.eval()
    
    # save trained parameters  
    state_dict = {
      'model': model.state_dict() if isinstance(model, torch.nn.Module) else [model_i.state_dict() for model_i in model],
      'optimizer': optimizer_rnn.state_dict() if isinstance(optimizer_rnn, torch.optim.Adam) else [optim_i.state_dict() for optim_i in optimizer_rnn],
      'groundtruth': {
        'alpha': alpha,
        'beta': beta,
        'forget_rate': forget_rate,
        'perseverance_bias': perseverance_bias,
        'alpha_penalty': alpha_penalty,
        'confirmation_bias': confirmation_bias,
      },
    }
    print('Finetuning finished.')
    print(f'Trained beta of RNN is: {model._beta_reward.item()} and {model._beta_choice.item()}')
    torch.save(state_dict, params_path.replace('_rnn_', '_rnn_finetuned_'))
    print(f'Saved RNN parameters to file {params_path}.')
    print(f'Finetuning took {time.time() - start_time:.2f} seconds.')
  
    # validate model
    print('\nTesting the finetuned RNN on the finetune dataset...')
    model.eval()
    with torch.no_grad():
      _, _, loss_test = rnn_training.fit_model(
          model=model,
          dataset_train=dataset_test,
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
    experiment = experiment_list_test[session_id]
    
    choices = experiment.choices
    rewards = experiment.rewards

    list_probs = []
    list_Qs = []
    list_qs = []
    list_hs = []

    # get q-values from groundtruth
    if data is None:
      qs_test, probs_test, _ = bandits.get_update_dynamics(experiment, agent_finetune)
      list_probs.append(np.expand_dims(probs_test, 0))
      list_Qs.append(np.expand_dims(qs_test[0], 0))
      list_qs.append(np.expand_dims(qs_test[1], 0))
      list_hs.append(np.expand_dims(qs_test[2], 0))
    elif np.mean(update_dynamics[0]) != -1:
      qs_test, probs_test = update_dynamics[1:], update_dynamics[0]
      list_probs.append(np.expand_dims(probs_test[idx_train:][session_id], 0))
      list_Qs.append(np.expand_dims(qs_test[0][idx_train:][session_id], 0))
      list_qs.append(np.expand_dims(qs_test[1][idx_train:][session_id], 0))
      list_hs.append(np.expand_dims(qs_test[2][idx_train:][session_id], 0))
      
    # get q-values from trained rnn
    qs_rnn, probs_rnn, _ = bandits.get_update_dynamics(experiment, rnn_agent)
    list_probs.append(np.expand_dims(probs_rnn, 0))
    list_Qs.append(np.expand_dims(qs_rnn[0], 0))
    list_qs.append(np.expand_dims(qs_rnn[1], 0))
    list_hs.append(np.expand_dims(qs_rnn[2], 0))

    # concatenate all choice probs and q-values
    probs = np.concatenate(list_probs, axis=0)
    Qs = np.concatenate(list_Qs, axis=0)
    qs = np.concatenate(list_qs, axis=0)
    hs = np.concatenate(list_hs, axis=0)

    labels = ['Ground Truth', 'RNN'][:Qs.shape[0]]
    colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']
    
    # normalize q-values
    def normalize(qs):
      return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

    # qs = normalize(qs)

    fig, axs = plt.subplots(5, 1, figsize=(20, 10))

    reward_probs = np.stack([experiment.reward_probabilities[:, i] for i in range(n_actions)], axis=0)
    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=reward_probs,
        timeseries_name='p(R)',
        labels=[f'Arm {a}' for a in range(n_actions)],
        color=['tab:purple', 'tab:cyan', 'tab:olive', 'tab:brown'],
        binary=True,
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
        binary=True,
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
  parser.add_argument('--perseverance_bias', type=float, default=0., help='perseverance bias')
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
    epochs_train=args.epochs,
    n_trials_per_session = args.n_trials_per_session,
    n_sessions = args.n_sessions,
    n_steps_per_call = args.n_steps_per_call,
    bagging = args.bagging,
    n_oversampling_train=args.n_oversampling,
    batch_size_train=args.batch_size,
    lr_train=args.learning_rate,

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
    perseverance_bias = args.perseverance_bias,
    alpha_penalty = args.alpha_p,
    confirmation_bias = args.confirmation_bias,
    # reward_update_rule = lambda q, reward: reward-q,
    
    # environment parameters
    sigma = args.sigma,
    non_binary_reward = args.non_binary_reward,
    
    analysis = args.analysis,
  )
  