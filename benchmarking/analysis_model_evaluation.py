import os
import sys

import numpy as np
import pickle
from tqdm import tqdm
from copy import copy, deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import get_scores_array
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_benchmark_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentQ, get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model


# data = 'data/sugawara2021_143_processed.csv'
# model_rnn = 'params/benchmarking/rnn_sugawara.pkl'
# model_benchmark = 'benchmarking/params/sugawara2021_143/non_hierarchical/traces.nc'

data = 'data/2arm/eckstein2022_291_processed.csv'
model_rnn = 'params/benchmarking/rnn_eckstein.pkl'
model_benchmark = 'benchmarking/params/eckstein2022_291/non_hierarchical/traces.nc'

models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']
# models = ['ApAnBcBr']

# setup rnn agent for comparison
agent_rnn = setup_agent_rnn(model_rnn)
n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

# setup sindy agent and get number of sindy coefficients which are not 0
agent_sindy = setup_agent_sindy(model_rnn, data)
n_parameters = agent_sindy._count_sindy_parameters(without_self=True)

# setup random, dummy AgentQ model (as quasi-baseline)
agent_rl = AgentQ(alpha=0.5, beta=5, perseverance_bias=1.0)

agent_mcmc = {}
for model in models:
    with open(model_benchmark.split('.')[0] + '_' + model + '.nc', 'rb') as file:
        mcmc = pickle.load(file)
    mcmc.print_summary()
    parameters = {
        'alpha_pos': 1,
        'alpha_neg': -1,
        'alpha_c': 1,
        'beta_c': 0,
        'beta_r': 1,
    }
    n_parameters_mcmc = 0
    # mcmc.print_summary()
    for p in mcmc.get_samples():
        parameters[p] = np.mean(mcmc.get_samples()[p], axis=0)
        n_parameters_mcmc += 1

    if np.mean(parameters['alpha_neg']) == -1:
        parameters['alpha_neg'] = parameters['alpha_pos']

    agent_mcmc[model] = (setup_benchmark_q_agent(parameters), n_parameters_mcmc)

# load data
experiment = convert_dataset(data)[1]
experiment_id = np.arange(0, len(experiment))

data = np.zeros((len(models)+3, 3))
n_experiments = len(experiment)

print('Get LL by RL-Baseline...')
df = get_scores_array(experiment_id, experiment, [agent_rl]*len(experiment), [2]*len(experiment))
data[0, :] = np.array((df['NLL'].sum(), df['AIC'].sum(), df['BIC'].sum()))

print('Get LL by RNN...')
df = get_scores_array(experiment_id, experiment, [agent_rnn]*len(experiment), [n_parameters_rnn]*len(experiment))
data[-2, :] = np.array((df['NLL'].sum(), df['AIC'].sum(), df['BIC'].sum()))

print('Get LL by SINDy...')
df = get_scores_array(experiment_id, experiment, [agent_sindy]*len(experiment), [agent_sindy._count_sindy_parameters(True)]*len(experiment))
data[-1, :] = np.array((df['NLL'].sum(), df['AIC'].sum(), df['BIC'].sum()))

for i in range(1, len(models)+1):
    key = list(agent_mcmc.keys())[i-1]
    print(f'Get LL by Benchmark ({key})...')
    df = get_scores_array(experiment_id, experiment, [agent_mcmc[key][0]]*len(experiment), [agent_mcmc[key][1]]*len(experiment))
    data[i, :] = np.array((df['NLL'].sum(), df['AIC'].sum(), df['BIC'].sum()))

import pandas as pd

df = pd.DataFrame(
    data=data,
    index=['RL']+models+['RNN', 'SINDy'],
    columns = ('NLL', 'AIC', 'BIC'),
    )

print(df)