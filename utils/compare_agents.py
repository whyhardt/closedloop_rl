import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import AgentQ, EnvironmentBanditsSwitch, plot_session, create_dataset, get_update_dynamics

agent1 = AgentQ(
    alpha=0.25,
    beta=3.,
    forget_rate=0.,
    perseverance_bias=0.,
    alpha_penalty=-1,
    confirmation_bias=0.,
    directed_exploration_bias=0.,
    undirected_exploration_bias=0.,
    )

agent2 = AgentQ(
    alpha=0.25,
    beta=3.,
    forget_rate=0.,
    perseverance_bias=0.,
    alpha_penalty=0.,
    confirmation_bias=0.,
    directed_exploration_bias=0.,
    undirected_exploration_bias=0.,
    )

env = EnvironmentBanditsSwitch(0.05, reward_prob_high=1.0, reward_prob_low=0.5)

colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

trajectory = create_dataset(agent1, env, 256, 1)[1][0]

values1, probs1, _ = get_update_dynamics(trajectory, agent1)
values2, probs2, _ = get_update_dynamics(trajectory, agent2)

choices = trajectory.choices
rewards = trajectory.rewards

list_probs = []
list_Qs = []
list_qs = []
list_hs = []
list_us = []
list_bs = []

list_probs.append(np.expand_dims(probs1, 0))
list_Qs.append(np.expand_dims(values1[0], 0))
list_qs.append(np.expand_dims(values1[1], 0))
list_hs.append(np.expand_dims(values1[2], 0))
list_us.append(np.expand_dims(values1[3], 0))
list_bs.append(np.expand_dims(values1[4], 0))
list_probs.append(np.expand_dims(probs2, 0))
list_Qs.append(np.expand_dims(values2[0], 0))
list_qs.append(np.expand_dims(values2[1], 0))
list_hs.append(np.expand_dims(values2[2], 0))
list_us.append(np.expand_dims(values2[3], 0))
list_bs.append(np.expand_dims(values2[4], 0))

probs = np.concatenate(list_probs, axis=0)
Qs = np.concatenate(list_Qs, axis=0)
qs = np.concatenate(list_qs, axis=0)
hs = np.concatenate(list_hs, axis=0)
us = np.concatenate(list_us, axis=0)
bs = np.concatenate(list_bs, axis=0)

fig, axs = plt.subplots(7, 1, figsize=(20, 10))

reward_probs = np.stack([trajectory.reward_probabilities[:, i] for i in range(agent1._n_actions)], axis=0)
plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=reward_probs,
    timeseries_name='Reward Probs',
    labels=[f'Arm {a}' for a in range(agent1._n_actions)],
    color=['tab:purple', 'tab:cyan'],
    binary=True,
    fig_ax=(fig, axs[0]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=probs[:, :, 0],
    timeseries_name='Choice Probs',
    color=colors,
    labels=['Agent 1', 'Agent 2'],
    binary=True,
    fig_ax=(fig, axs[1]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=Qs[:, :, 0],
    timeseries_name='Q Arm 0',
    color=colors,
    binary=True,
    fig_ax=(fig, axs[2]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=qs[:, :, 0],
    timeseries_name='q Arm 0',
    color=colors,
    binary=True,
    fig_ax=(fig, axs[3]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=hs[:, :, 0],
    timeseries_name='h Arm 0',
    color=colors,
    binary=True,
    fig_ax=(fig, axs[4]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=us[:, :, 0],
    timeseries_name='u Arm 0',
    color=colors,
    binary=True,
    fig_ax=(fig, axs[5]),
    )

plot_session(
    compare=True,
    choices=choices,
    rewards=rewards,
    timeseries=bs[:, :, 0],
    timeseries_name='beta',
    color=colors,
    binary=True,
    fig_ax=(fig, axs[6]),
    )

plt.show()