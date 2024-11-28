import os
import sys

import numpy as np
import pickle
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import akaike_information_criterion, bayesian_information_criterion, log_likelihood
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_custom_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentQ, get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model


data = 'data/2arm/sugawara2021_143_processed.csv'

model_rnn = 'params/benchmarking/gru_super_sugawara.pkl'
model_benchmark = 'benchmarking/params/traces_MODEL.nc'

# models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr']
models = ['ApAnBcBr']

# setup rnn agent for comparison
# agent_rnn = setup_agent_rnn(model_rnn)
# n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

# setup sindy agent and get number of sindy coefficients which are not 0
# agent_sindy = setup_agent_sindy(model_rnn, data, hidden_size=16)
# n_parameters = agent_sindy._count_sindy_parameters(without_self=True)

# setup random, dummy AgentQ model (as quasi-baseline)
agent_rl = AgentQ()

agent_mcmc = {}
for model in models:
    with open(model_benchmark.replace('MODEL', model), 'rb') as file:
        mcmc = pickle.load(file)
        
    parameters = {
        'alpha_pos': 1,
        'alpha_neg': -1,
        'alpha_c': 1,
        'beta_c': 0,
        'beta_r': 1,
    }
    n_parameters_mcmc = 0
    for p in mcmc.get_samples():
        parameters[p] = np.mean(mcmc.get_samples()[p], axis=0)
        n_parameters_mcmc += 1
    # n_parameters_mcmc = np.full((len(job_id)), n_parameters_mcmc)

    if np.mean(parameters['alpha_neg']) == -1:
        parameters['alpha_neg'] = parameters['alpha_pos']

    def update_rule(Q, C, a, r):
        ch = np.eye(2)[a]
        
        # Compute prediction errors for each outcome
        rpe = (r - Q) * ch
        cpe = ch - C
        
        # Update values
        lr = np.where(r > 0.5, parameters['alpha_pos'], parameters['alpha_neg'])
        Q = Q + lr * rpe
        C = C + parameters['alpha_c'] * cpe
        
        return Q, C

    def get_q(Q, C):
        return parameters['beta_r'] * Q + parameters['beta_c'] * C

    agent_mcmc[model] = (setup_custom_q_agent(update_rule, get_q), n_parameters_mcmc)

# load data
_, experiment_list, _, _ = convert_dataset(data)

def get_scores(agent, n_parameters) -> float:
    aic = 0
    bic = 0
    nll = 0
    for experiment in tqdm(experiment_list):
        probs = get_update_dynamics(experiment, agent)[1]
        ll = log_likelihood(experiment.choices, probs)
        bic += bayesian_information_criterion(experiment.choices, probs[:, 0], n_parameters, ll)
        aic += akaike_information_criterion(experiment.choices, probs[:, 0], n_parameters, ll)
        nll -= ll
    return nll, aic, bic

nll_mcmc, aic_mcmc, bic_mcmc = np.zeros((len(models))), np.zeros((len(models))), np.zeros((len(models)))
nll_rl, aic_rl, bic_rl = get_scores(agent_rl, 2)
for i, mcmc in enumerate(agent_mcmc):
    nll_mcmc[i], aic_mcmc[i], bic_mcmc[i] = get_scores(agent_mcmc[mcmc][0], agent_mcmc[mcmc][1])
# nll_rnn, aic_rnn, bic_rnn = get_scores(agent_rnn, n_parameters_rnn)
# nll_sindy, aic_sindy, bic_sindy = get_scores(agent_sindy, agent_sindy._count_sindy_parameters(True))

import pandas as pd

benchmark_values = [
    [nll_rl, aic_rl, bic_rl],
    # [nll_rnn, aic_rnn, bic_rnn],
]
for i in range(len(models)):
    benchmark_values.append([nll_mcmc[i], aic_mcmc[i], bic_mcmc[i]])

df = pd.DataFrame(
    benchmark_values,
    index = (
        'RL-Baseline',
        # 'RNN', 
        'RL-Benchmark',  
        # 'SINDy',
        ),
    columns = ('NLL', 'AIC', 'BIC'),
    )

print(df)
