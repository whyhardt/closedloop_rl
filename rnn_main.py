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
from utils import convert_dataset, plotting

def main(
  checkpoint = False,
  model: str = None,
  data: str = None,

  # rnn parameters
  hidden_size = 8,
  dropout = 0.1,
  participant_emb = False,

  # data and training parameters
  epochs = 128,
  train_test_ratio = 0.7,
  n_trials_per_session = 64,
  n_sessions = 4096,
  bagging = False,
  sequence_length = 32,
  n_steps_per_call = 16,  # -1 for full sequence
  batch_size = -1,  # -1 for one batch per epoch
  learning_rate = 1e-2,
  convergence_threshold = 1e-6,
  weight_decay = 0.,
  parameter_variance = 0.,
  
  # ground truth parameters
  beta_reward = 3.,
  alpha = 0.25,
  alpha_penalty = -1.,
  alpha_counterfactual = 0.,
  beta_choice = 0.,
  alpha_choice = 1.,
  forget_rate = 0.,
  confirmation_bias = 0.,
  reward_prediction_error: Callable = None,
  
  # environment parameters
  n_actions = 2,
  sigma = 0.1,
  counterfactual = False,
  
  analysis: bool = False,
  session_id: int = 0
  ):

  session_id_test = session_id
  
  if not os.path.exists('params'):
    os.makedirs('params')
  
  # tracked variables in the RNN
  x_train_list = ['xQf', 'xLR', 'xLR_cf', 'xC', 'xCf']
  control_list = ['ca', 'cr', 'cp', 'ca_repeat', 'cQ']
  sindy_feature_list = x_train_list + control_list

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  dataset_test = None
  if data is None:
    print('No path to dataset provided.')
    
    # setup
    environment = bandits.EnvironmentBanditsDrift(sigma, n_actions, counterfactual=counterfactual)
    # environment = bandits.EnvironmentBanditsSwitch(sigma, counterfactual=counterfactual)
    agent = bandits.AgentQ(
      n_actions=n_actions, 
      alpha_reward=alpha, 
      beta_reward=beta_reward, 
      forget_rate=forget_rate, 
      beta_choice=beta_choice, 
      alpha_choice=alpha_choice,
      alpha_penalty=alpha_penalty, 
      alpha_counterfactual=alpha_counterfactual, 
      confirmation_bias=confirmation_bias, 
      parameter_variance=parameter_variance,
      )
    if reward_prediction_error is not None:
      agent.set_reward_prediction_error(reward_prediction_error)
    print('Setup of the environment and agent complete.')
    
    print('Generating the synthetic dataset...')
    dataset, _, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions,
        sample_parameters=parameter_variance!=0,
        sequence_length=sequence_length,
        device=device)
    
    if train_test_ratio == 0:
      dataset_test, experiment_list, _ = bandits.create_dataset(
        agent=agent,
        environment=environment,
        n_trials_per_session=256,
        n_sessions=n_sessions,
        sample_parameters=parameter_variance!=0,
        device=device)

    print('Generation of dataset complete.')
  else:
    dataset, experiment_list, _, _ = convert_dataset.convert_dataset(data, sequence_length=sequence_length)
    dataset_test = rnn_utils.DatasetRNN(dataset.xs, dataset.ys)
    
    # # check if groundtruth parameters in data - only applicable to generated data with e.g. utils/create_dataset.py
    # if 'mean_beta' in df.columns:
    #   beta = df['beta'].values[idx_train]
    #   alpha = df['alpha'].values[idx_train]
    #   alpha_penalty = df['alpha_penalty'].values[idx_train]
    #   confirmation_bias = df['confirmation_bias'].values[idx_train]
    #   forget_rate = df['forget_rate'].values[idx_train]
    #   perseverance_bias = df['perseverance_bias'].values[idx_train]
  
  n_participants = len(experiment_list)
  
  if train_test_ratio > 0:
    # setup of training and test dataset
    index_train = int(train_test_ratio * dataset.xs.shape[1])
    
    xs_test, ys_test = dataset.xs[:, index_train:], dataset.ys[:, index_train:]
    xs_train, ys_train = dataset.xs[:, :index_train], dataset.ys[:, :index_train]
    dataset_train = bandits.DatasetRNN(xs_train, ys_train, sequence_length=sequence_length)
    if dataset_test is None:
      dataset_test = bandits.DatasetRNN(xs_test, ys_test)  
  else:
    if dataset_test is None:
      dataset_test = dataset
    dataset_train = bandits.DatasetRNN(dataset.xs, dataset.ys, sequence_length=sequence_length)
    
  experiment_test = experiment_list[session_id_test][-dataset_test.xs.shape[1]:]
    
  if model is None:
    params_path = rnn_utils.parameter_file_naming(
      'params/params', 
      alpha=alpha, 
      beta_reward=beta_reward, 
      alpha_counterfactual=alpha_counterfactual,
      forget_rate=forget_rate, 
      beta_choice=beta_choice,
      alpha_choice=alpha_choice,
      alpha_penalty=alpha_penalty,
      confirmation_bias=confirmation_bias, 
      variance=parameter_variance, 
      verbose=True,
      )
  else:
    params_path = model

  # define model
  model = rnn.RLRNN(
      n_actions=n_actions, 
      hidden_size=hidden_size, 
      init_value=0.5,
      device=device,
      list_sindy_signals=sindy_feature_list,
      dropout=dropout,
      n_participants=n_participants,
      participant_emb=participant_emb,
      counterfactual=dataset_train.xs[:, :, n_actions+1].mean() != -1,
      ).to(device)

  optimizer_rnn = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

  print('Setup of the RNN model complete.')

  if checkpoint:
      model, optimizer_rnn = rnn_utils.load_checkpoint(params_path, model, optimizer_rnn)
      print('Loaded model parameters.')

  loss_test = None
  if epochs > 0:
    start_time = time.time()
    
    #Fit the RNN
    print('Training the RNN...')
    model, optimizer_rnn, _ = rnn_training.fit_model(
        model=model,
        dataset_train=dataset_train,
        optimizer=optimizer_rnn,
        convergence_threshold=convergence_threshold,
        epochs=epochs,
        batch_size=batch_size,
        bagging=bagging,
        n_steps_per_call=n_steps_per_call,
    )
    
    # save trained parameters
    state_dict = {'model': model.state_dict(), 'optimizer': optimizer_rnn.state_dict()}
    
    print('Training finished.')
    torch.save(state_dict, params_path)
    print(f'Saved RNN parameters to file {params_path}.')
    print(f'Training took {time.time() - start_time:.2f} seconds.')
  else:
    if isinstance(model, list):
      model = model[0]
      optimizer_rnn = optimizer_rnn[0]
  
  # validate model
  if dataset_test is not None:
    print('\nTesting the trained RNN on the test dataset...')
    model.eval()
    with torch.no_grad():
      _, _, loss_test = rnn_training.fit_model(
          model=model,
          dataset_train=dataset_train,
      )
  
  # -----------------------------------------------------------
  # Analysis
  # -----------------------------------------------------------
  
  if analysis:
    # print(f'Betas of model: {(model._beta_reward.item(), model._beta_choice.item())}')
    # Synthesize a dataset using the fitted network
    model.set_device(torch.device('cpu'))
    model.to(torch.device('cpu'))
    agent_rnn = bandits.AgentNetwork(model, n_actions=n_actions, deterministic=True)
    
    # get analysis plot
    if data is None:
      agents = {'groundtruth': agent, 'rnn': agent_rnn}
    else:
      agents = {'rnn': agent_rnn}

    fig, axs = plotting.plot_session(agents, experiment_test)
    
    fig.suptitle('$beta_r=$'+str(np.round(agent_rnn._beta_reward, 2)) + '; $beta_c=$'+str(np.round(agent_rnn._beta_choice, 2)))
    plt.show()
    
  return model, loss_test


if __name__=='__main__':
  
  parser = argparse.ArgumentParser(description='Trains the RNN on behavioral data to uncover the underlying Q-Values via different cognitive mechanisms.')
  
  # Training parameters
  parser.add_argument('--checkpoint', action='store_true', help='Whether to load a checkpoint')
  parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
  parser.add_argument('--data', type=str, default=None, help='Path to dataset')
  parser.add_argument('--n_actions', type=int, default=2, help='Number of possible actions')
  parser.add_argument('--epochs_train', type=int, default=128, help='Number of epochs for training')
  parser.add_argument('--n_trials_per_session', type=int, default=64, help='Number of trials per session')
  parser.add_argument('--n_sessions', type=int, default=4096, help='Number of sessions')
  parser.add_argument('--n_steps_per_call', type=int, default=8, help='Number of steps per call')
  parser.add_argument('--bagging', action='store_true', help='Whether to use bagging')
  parser.add_argument('--batch_size_train', type=int, default=-1, help='Batch size')
  parser.add_argument('--lr_train', type=float, default=0.01, help='Learning rate of the RNN')

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
    data = args.data,
    n_actions=args.n_actions,

    # training parameters
    epochs=args.epochs_train,
    n_trials_per_session = args.n_trials_per_session,
    n_sessions = args.n_sessions,
    n_steps_per_call = args.n_steps_per_call,
    bagging = args.bagging,
    batch_size=args.batch_size_train,
    learning_rate=args.lr_train,

    # rnn parameters
    hidden_size = args.hidden_size,
    dropout = args.dropout,
    
    # ground truth parameters
    alpha = args.alpha,
    beta_reward = args.beta,
    forget_rate = args.forget_rate,
    beta_choice = args.perseverance_bias,
    alpha_penalty = args.alpha_p,
    confirmation_bias = args.confirmation_bias,

    # environment parameters
    sigma = args.sigma,
    
    analysis = args.analysis,
  )
  