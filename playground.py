import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

from resources.bandits import AgentQ, get_update_dynamics
from resources.rnn_utils import split_data_along_sessiondim
from utils.convert_dataset import convert_dataset
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.plotting import plot_session
from utils.convert_dataset import convert_dataset
from resources import rnn, sindy_utils, rnn_utils
from resources.model_evaluation import log_likelihood

from benchmarking.benchmarking_lstm import setup_agent_lstm
from benchmarking import benchmarking_eckstein2022, benchmarking_dezfouli2019

study = 'eckstein2022'

path_data = f'data/{study}/{study}.csv'
path_rnn = f'params/{study}/rnn_{study}_l2_0_001.pkl'

participant_id = 0 # 0, 150, 289

class_rnn = rnn.RLRNN_eckstein2022
sindy_config = sindy_utils.SindyConfig_eckstein2022
additional_inputs = None

# dataset = convert_dataset(path_data)[0]
# dataset = sindy_utils.generate_off_policy_data(
#     participant_id=participant_id,
#     block=0,
#     experiment_id=0,
#     additional_inputs=torch.zeros(0),
#     n_trials_off_policy=20,
#     n_trials_same_action_off_policy=5,
#     n_sessions_off_policy=1,
#     sigma_drift=0.5,
# )

rewards_int = [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0]
choices_int = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
rewards = torch.zeros((20, 2)) - 1
for i, choice in enumerate(choices_int):
    rewards[i, choice] = rewards_int[i]
choices = torch.eye(2)[choices_int]
blocks = torch.zeros((20, 1))
experiment_id = torch.zeros((20, 1))
participant_ids = torch.zeros((20, 1)) + participant_id
xs = torch.concat((choices, rewards, blocks, experiment_id, participant_ids), dim=-1).reshape(1, 20, -1)
dataset = rnn_utils.DatasetRNN(xs, choices)

# agent_gql, n_parameters = setup_agent_gql(path_model=path_gql, model_config='PhiChiBetaKappaC', dimensions=2)

# agent_mcmc, n_parameters = setup_agent_mcmc_dezfouli(path_mcmc)

agent_rnn = setup_agent_rnn(
    class_rnn=rnn.RLRNN_eckstein2022,
    path_model=path_rnn,
)
# probs_rnn = get_update_dynamics(experiment=dataset.xs[participant_id].numpy(), agent=agent_rnn)[1]
# ll_rnn = log_likelihood(dataset.xs[participant_id, :len(probs_rnn), :agent_rnn._n_actions].numpy(), probs_rnn)
# lik_rnn = np.exp(ll_rnn / len(probs_rnn))
# print(f"Avg. Trial Likelihood RNN: {lik_rnn:.5f}")

# agent_rnn_2 = setup_agent_rnn(
#     class_rnn=rnn.RLRNN_eckstein2022_FC,
#     path_model=path_rnn_2,
# )

agent_spice = setup_agent_spice(
    class_rnn=rnn.RLRNN_eckstein2022,
    path_rnn=path_rnn,
    path_spice=path_rnn.replace('rnn', 'spice'),
)
agent_spice.new_sess()
# probs_spice = get_update_dynamics(experiment=dataset.xs[participant_id].numpy(), agent=agent_spice)[1]
# ll_spice = log_likelihood(dataset.xs[participant_id, :len(probs_spice), :agent_spice._n_actions].numpy(), probs_spice)
# lik_spice = np.exp(ll_spice / len(probs_spice))
# print(f"Avg. Trial Likelihood SPICE: {lik_spice:.5f}")

print('\n\nDiscovered SPICE models:\n')
for pid in [participant_id]:#, 20, 40, 60, 80, 100]:
    print(f'For participant {pid}:\n')
    agent_spice.print_model(participant_id=pid)
    for value in agent_spice.get_betas():
        print(f"{value}: {agent_spice.get_betas()[value]}")

fig, axs = plot_session(
    agents={
        'rnn': agent_rnn,
        # 'benchmark': agent_rnn_2,
        'sindy': agent_spice,
        # 'benchmark': agent_gql[participant_id],
        },
    experiment=dataset.xs[0],
    display_choice=0
    )

# plt.show()

plt.savefig(f'participant{participant_id}.png', dpi=500)