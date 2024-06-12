import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

import pysindy as ps

sys.path.append('resources')  # add source directoy to path
from resources.rnn import RLRNN
from resources.bandits import AgentQ, AgentNetwork, AgentSindy, EnvironmentBanditsDrift, plot_session, get_update_dynamics, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, make_sindy_data
from resources.rnn_utils import parameter_file_naming

warnings.filterwarnings("ignore")

# sindy parameters
threshold = 0.03
polynomial_degree = 2
regularization = 1e-1
ensemble = False
library_ensemble = False
library = ps.PolynomialLibrary(degree=polynomial_degree)

# training dataset parameters
n_trials_per_session = 200
n_sessions = 10

# ground truth parameters
gen_alpha = .25
gen_beta = 3
forget_rate = 0.
perseverance_bias = 0.

# environment parameters
non_binary_reward = False
n_actions = 2
sigma = .1

# rnn parameters
hidden_size = 4
last_output = False
last_state = False
use_habit = True
use_lstm = False

# set up ground truth agent and environment
environment = EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_rewards=non_binary_reward)
agent = AgentQ(gen_alpha, gen_beta, n_actions, forget_rate, perseverance_bias)
dataset_test, experiment_list_test = create_dataset_bandits(agent, environment, n_trials_per_session, 1)

# set up rnn agent and expose q-values to train sindy
params_path = parameter_file_naming('params/params', use_lstm, last_output, last_state, use_habit, gen_beta, forget_rate, perseverance_bias, non_binary_reward, verbose=True)
rnn = RLRNN(n_actions, hidden_size, 0.5, use_habit, last_output, last_state)
rnn.load_state_dict(torch.load(params_path, map_location=torch.device('cpu'))['model'])
agent_rnn = AgentNetwork(rnn, n_actions, use_habit)

# create dataset for sindy training
x_train, control, feature_names = create_dataset(agent_rnn, environment, n_trials_per_session, n_sessions, normalize=False)

# train one sindy model per x_train variable instead of one sindy model for all
sindy_models = []
for i in range(x_train[0].shape[-1]):
    if 'x' in feature_names[i]:
        x_train_i = [x_sample[:, i].reshape(-1, 1) for x_sample in x_train]
        feature_names_i = [feature_names[i]] + feature_names[x_train[0].shape[-1]:]
        sindy_models.append(ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold, verbose=True, alpha=regularization),
            feature_library=library,
            discrete_time=True,
            feature_names=feature_names_i,
        ))
        
        sindy_models[-1].fit(x_train_i, u=control, t=1, multiple_trajectories=True, ensemble=ensemble, library_ensemble=library_ensemble)
        sindy_models[-1].print()
        
# mimic behavior of rnn with sindy
def update_rule_sindy(q, h, choice, prev_choice, reward):
    if choice == 0:
        # blind update
        q = sindy_models[0].simulate(q, t=2, u=np.array([choice, prev_choice, reward]).reshape(1, control[0].shape[-1]))[-1]
    elif choice == 1:
        # reward-based update
        q = sindy_models[1].simulate(q, t=2, u=np.array([choice, prev_choice, reward]).reshape(1, control[0].shape[-1]))[-1]
    # add habit (perseverance bias)
    if prev_choice != np.nan:
        h = sindy_models[2].simulate(h, t=2, u=np.array([choice, prev_choice, reward]).reshape(1, control[0].shape[-1]))[-1]
    return q, h

# classic way to generate data for sindy training and fit sindy
# dataset_rnn, experiment_list_rnn = create_dataset_bandits(agent_rnn, environment, n_trials_per_session, n_sessions)
# x_train, control, feature_names = make_sindy_data(experiment_list_rnn, agent_rnn)

# sindy = ps.SINDy(
#         optimizer=ps.STLSQ(threshold=threshold, verbose=True, alpha=regularization),
#         feature_library=library,
#         discrete_time=True,
#         feature_names=feature_names,
#     )
# sindy.fit(x_train, t=1, u=control, ensemble=ensemble, library_ensemble=library_ensemble, multiple_trajectories=True)
# sindy.print()

# update_rule_sindy = lambda q, choice, reward: sindy.simulate(q, t=2, u=np.array([choice, reward]).reshape(1, 2))[-1]

agent_sindy = AgentSindy(n_actions)
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
qs = (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

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

