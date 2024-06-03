import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt

import pysindy as ps

sys.path.append('resources')  # add source directoy to path
from resources.rnn import HybRNN
from resources.bandits import AgentQ, AgentNetwork, AgentSindy, EnvironmentBanditsDrift, create_dataset, plot_session
from resources.sindy_utils import get_q, make_sindy_data
from resources.rnn_utils import parameter_file_naming

warnings.filterwarnings("ignore")

# sindy parameters
threshold = 0.015
polynomial_degree = 2
ensemble = False
library_ensemble = False

# training dataset parameters
n_trials_per_session = 200
n_sessions = 16

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
dataset_test, experiment_list_test = create_dataset(agent, environment, n_trials_per_session, 1)

# set up rnn agent
params_path = parameter_file_naming('params/params', use_lstm, last_output, last_state, use_habit, gen_beta, forget_rate, perseverance_bias, non_binary_reward, verbose=True)
rnn = HybRNN(n_actions, hidden_size, 0.5, use_habit, last_output, last_state)
rnn.load_state_dict(torch.load(params_path, map_location=torch.device('cpu'))['model'])
agent_rnn = AgentNetwork(rnn, n_actions, use_habit)

# create dataset for sindy training
dataset_rnn, experiment_list_rnn = create_dataset(agent_rnn, environment, n_trials_per_session, n_sessions)
x_train, control, feature_names = make_sindy_data(experiment_list_rnn, agent_rnn)

# set up sindy agent
library = ps.PolynomialLibrary(degree=polynomial_degree)
sindy = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, verbose=True, alpha=0.1),
        feature_library=library,
        discrete_time=True,
        feature_names=feature_names,
    )
sindy.fit(x_train, t=1, u=control, ensemble=ensemble, library_ensemble=library_ensemble, multiple_trajectories=True)
sindy.print()
# update_rule_rnnsindy = lambda q, choice, reward: sindy.simulate(q, t=2, u=np.array([choice, reward]).reshape(1, 2))[-1]

def update_rule_sindy(qr, qf, h, choice, reward):
    h, qr, qf = sindy.simulate(np.array([h, qr, qf]).reshape(1, -1), t=2, u=np.array([choice, reward]).reshape(1, -1))[-1]
    return 

agent_sindy = AgentSindy(alpha=0, beta=1, n_actions=n_actions)
agent_sindy.set_update_rule(update_rule_sindy)

# Analysis
labels = ['Ground Truth', 'RNN', 'SINDy']
experiment_test = experiment_list_test[0]
choices = experiment_test.choices
rewards = experiment_test.rewards

list_probs = []
list_qs = []

# get q-values from groundtruth
qs_test, probs_test = get_q(experiment_test, agent)
list_probs.append(np.expand_dims(probs_test, 0))
list_qs.append(np.expand_dims(qs_test, 0))

# get q-values from trained rnn
qs_rnn, probs_rnn = get_q(experiment_test, agent_rnn)
list_probs.append(np.expand_dims(probs_rnn, 0))
list_qs.append(np.expand_dims(qs_rnn, 0))

# get q-values from trained sindy
qs_sindy, probs_sindy = get_q(experiment_test, agent_sindy)
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

