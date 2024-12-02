import os
import sys

import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import get_scores_array
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_benchmark_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentQ, get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model


data = 'data/sugawara2021_143_processed.csv'
# data = 'data/2arm/eckstein2022_291_processed.csv'

model_benchmark = 'benchmarking/params/sugawara2021_143/non_hierarchical/traces.nc'

model = 'ApAcBcBr'

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

agent_mcmc = setup_benchmark_q_agent(parameters)

# load data
experiment = convert_dataset(data)[1]

get_scores_array([agent_mcmc]*len(experiment), [n_parameters_mcmc]*len(experiment), verbose=True)