import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import copy

import pysindy as ps

sys.path.append('resources')  # add source directoy to path
from resources.rnn import RLRNN, EnsembleRNN
from resources.bandits import AgentQ, AgentNetwork, AgentSindy, EnvironmentBanditsDrift, plot_session, get_update_dynamics, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, check_library_setup, remove_control_features, extract_samples
from resources.rnn_utils import parameter_file_naming

warnings.filterwarnings("ignore")

# Implementing ensemble rnn did not help in terms of interchangable value updates at different stages 
# TODO: limit the SINDy library to only the relevant variables actually used in the RNN at the respective stage

# sindy parameters
threshold = 0.03
polynomial_degree = 1
regularization = 1e0
sindy_ensemble = False
library_ensemble = False
library = ps.PolynomialLibrary(degree=polynomial_degree)

# training dataset parameters
n_trials_per_session = 200
n_sessions = 10

# ground truth parameters
gen_alpha = .25
gen_beta = 3
forget_rate = 0.1
perseverance_bias = 0.

# environment parameters
non_binary_reward = False
n_actions = 2
sigma = .1

# rnn parameters
hidden_size = 4
last_output = False
last_state = False
use_habit = False
use_lstm = False
voting_type = EnsembleRNN.MEDIAN

# tracked variables in the RNN
x_train_list = ['xQf','xQr']
control_list = ['ca','ca[k-1]', 'cr']
if use_habit:
  x_train_list += ['xH']
sindy_feature_list = x_train_list + control_list

# data-filter setup aka which samples are allowed as training samples in each SINDy model corresponding to the given filter condition
# key is the SINDy submodel name, value is a list with the first element being the feature name to be used as a filter and the second element being the filter condition
# Example:
# 'xQf': ['ca', 0] means that only samples where the feature 'ca' is 0 are used for training the SINDy model 'xQf'
datafilter_setup = {
    'xQf': ['ca', 0],
    'xQr': ['ca', 1],
}
# library setup aka which terms are allowed as control inputs in each SINDy model
# key is the SINDy submodel name, value is a list of
library_setup = {
    'xQf': [],
    'xQr': ['cr'],
}
if use_habit:
    library_setup['xH'] = ['ca[k-1]']
if not check_library_setup(library_setup, sindy_feature_list, verbose=False):
    raise ValueError('Library setup does not match feature list.')

# set up ground truth agent and environment
environment = EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_rewards=non_binary_reward)
agent = AgentQ(gen_alpha, gen_beta, n_actions, forget_rate, perseverance_bias)
dataset_test, experiment_list_test = create_dataset_bandits(agent, environment, n_trials_per_session, 1)

# set up rnn agent and expose q-values to train sindy
params_path = parameter_file_naming('params/params', use_lstm, last_output, last_state, use_habit, gen_beta, forget_rate, perseverance_bias, non_binary_reward, verbose=True)
state_dict = torch.load(params_path, map_location=torch.device('cpu'))['model']
rnn = RLRNN(n_actions, hidden_size, 0.5, use_habit, last_output, last_state, sindy_feature_list)
if isinstance(state_dict, dict):
    rnn.load_state_dict(state_dict)
elif isinstance(state_dict, list):
    print('Loading ensemble model...')
    model_list = []
    for state_dict_i in state_dict:
        model_list.append(copy(rnn))
        model_list[-1].load_state_dict(state_dict_i)
    rnn = EnsembleRNN(model_list, voting_type=voting_type)
agent_rnn = AgentNetwork(rnn, n_actions, use_habit)

# create dataset for sindy training
x_train, control, feature_names = create_dataset(agent_rnn, environment, n_trials_per_session, n_sessions, normalize=True)

# train one sindy model per x_train variable instead of one sindy model for all
sindy_models = {key: None for key in library_setup.keys()}
for i in range(x_train[0].shape[-1]):
    print('\n')
    x_train_i = [x_sample[:, i].reshape(-1, 1) for x_sample in x_train]
    feature_names_i = [feature_names[i]] + feature_names[x_train[0].shape[-1]:]
    if feature_names[i] in datafilter_setup:
        x_train_i, control_i, feature_names_i = extract_samples(x_train_i, control, feature_names_i, datafilter_setup[feature_names[i]][0], datafilter_setup[feature_names[i]][1])
    else:
        control_i = control
    control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[feature_names[i]])
    feature_names_i = [feature_names[i]] + library_setup[feature_names[i]]
    if control_i is None:
        control_i = [np.zeros_like(x_train_i[0]) for _ in range(len(x_train_i))]
        feature_names_i = feature_names_i + ['u']
    # feature_names_i = [feature_names[i]] + feature_names[x_train[0].shape[-1]:]
    sindy_models[feature_names[i]] = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, verbose=True, alpha=regularization),
        feature_library=library,
        discrete_time=True,
        feature_names=feature_names_i,
    )
    
    sindy_models[feature_names[i]].fit(x_train_i, u=control_i, t=1, multiple_trajectories=True, ensemble=sindy_ensemble, library_ensemble=library_ensemble)
    print(f'\nSINDy model for {feature_names[i]}:')
    sindy_models[feature_names[i]].print()
    
# mimic behavior of rnn with sindy
def update_rule_sindy(q, h, choice, prev_choice, reward):
    if choice == 0:
        # blind update
        q = sindy_models['xQf'].simulate(q, t=2, u=np.array([0]).reshape(1, 1))[-1]
    elif choice == 1:
        # reward-based update
        q = sindy_models['xQr'].simulate(q, t=2, u=np.array([reward]).reshape(1, 1))[-1]
    # add habit (perseverance bias)
    if use_habit and prev_choice != np.nan:
        h = sindy_models['xH'].simulate(h, t=2, u=np.array([prev_choice]).reshape(1, 1))[-1]
    return q, h

# get trained beta from RNN
if isinstance(rnn, RLRNN):
    beta_rnn = rnn.beta.item()
elif isinstance(rnn, EnsembleRNN):
    beta_rnn = rnn.vote(torch.tensor([model_i.beta for model_i in rnn]).reshape(1, -1), rnn.voting_type).item()
print(f'Trained beta from RNN: {np.round(beta_rnn, 2)}')

agent_sindy = AgentSindy(n_actions, beta_rnn)
agent_sindy.set_update_rule(update_rule_sindy)

# test sindy agent
dataset_sindy, experiment_list_sindy = create_dataset_bandits(agent_sindy, environment, n_trials_per_session, 1)

# Analysis
labels = ['Ground Truth', 'RNN', 'SINDy']
experiment_test = experiment_list_test[0]
choices = experiment_test.choices
rewards = experiment_test.rewards

list_probs = []
list_qs = []

# get q-values from groundtruth
qs_test, probs_test = get_update_dynamics(experiment_test, agent)
list_probs.append(np.expand_dims(probs_test, 0))
list_qs.append(np.expand_dims(qs_test, 0))

# get q-values from trained rnn
qs_rnn, probs_rnn = get_update_dynamics(experiment_test, agent_rnn)
list_probs.append(np.expand_dims(probs_rnn, 0))
list_qs.append(np.expand_dims(qs_rnn, 0))

# get q-values from trained sindy
qs_sindy, probs_sindy = get_update_dynamics(experiment_test, agent_sindy)
list_probs.append(np.expand_dims(probs_sindy, 0))
list_qs.append(np.expand_dims(qs_sindy, 0))

colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

# concatenate all choice probs and q-values
probs = np.concatenate(list_probs, axis=0)
qs = np.concatenate(list_qs, axis=0)

# normalize q-values
# qs = (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

fig, axs = plt.subplots(4, 1, figsize=(20, 10))

reward_probs = np.stack([experiment_test.timeseries[:, i] for i in range(n_actions)], axis=0)
plot_session(
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

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=probs[:, :, 0],
    timeseries_name='Choice Probs',
    color=colors,
    labels=labels,
    binary=not non_binary_reward,
    fig_ax=(fig, axs[1]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=qs[:, :, 0],
    timeseries_name='Q-Values',
    color=colors,
    binary=not non_binary_reward,
    fig_ax=(fig, axs[2]),
    )

dqs_arms = np.diff(qs, axis=2)

plot_session(
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

