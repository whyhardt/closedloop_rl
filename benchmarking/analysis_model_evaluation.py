import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import akaike_information_criterion, bayesian_information_criterion, log_likelihood
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_custom_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import AgentQ, get_update_dynamics


data = 'data/sugawara2021_143_processed.csv'

model_rnn = 'params/benchmarking/sugawara2021_143_3.pkl'
model_benchmark = None

# setup rnn agent for comparison
agent_rnn = setup_agent_rnn(model_rnn)
n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

# setup sindy agent and get number of sindy coefficients which are not 0
agent_sindy = setup_agent_sindy(model_rnn, data)
n_parameters = agent_sindy._count_sindy_parameters(without_self=True)

# setup random, dummy AgentQ model (as quasi-baseline)
agent_rl = AgentQ()

# setup optimized AgentQ from paper (parameters from paper/supplementary/table-S3)
# agent_rl_bm = AgentQ(alpha=0.36, beta=0.32)
parameters = {
    'alpha': 0.36,
    'beta': 0.32,
    'alphaC': 0.,
    'betaC': 0.,
}

def update_rule(Q, C, a, r):
    Q[a] = Q[a] + parameters['alpha'] * (r - Q[a])
    return Q, C

def get_choice_probs(Q, C):
    p1 = 1 / (1 + np.exp(-parameters['beta'] * (Q[0] - Q[1])))
    return np.array((p1, 1-p1))

agent_rl_bm = setup_custom_q_agent(update_rule, get_choice_probs)


# setup custom AgentQ with perseverance (parameters from paper/supplementary/table-S3)
parameters = {
    'alpha': 0.45,
    'beta': 0.19,
    'alphaC': 0.41,
    'betaC': 1.10,
}

def update_rule(Q, C, a, r):
    Q[a] = Q[a] + parameters['alpha'] * (r - Q[a])
    C[0] = C[0] + parameters['alphaC'] * ((1 if a == 0 else -1) - C[0])
    return Q, C

def get_choice_probs(Q, C):
    p1 = 1 / (1 + np.exp(-(parameters['beta'] * (Q[0] - Q[1]) + parameters['betaC'] * C[0])))
    return np.array((p1, 1-p1))

agent_benchmark = setup_custom_q_agent(update_rule, get_choice_probs)

# load data
_, experiment_list, _, _ = convert_dataset(data)

def get_scores(agent, n_parameters) -> float:
    aic = 0
    bic = 0
    nll = 0
    for experiment in experiment_list:
        probs = get_update_dynamics(experiment, agent)[1]
        ll = log_likelihood(experiment.choices, probs[:, 0])
        bic += bayesian_information_criterion(experiment.choices, probs[:, 0], n_parameters, ll)
        aic += akaike_information_criterion(experiment.choices, probs[:, 0], n_parameters, ll)
        nll -= ll
    return nll, aic, bic

nll_rl, aic_rl, bic_rl = get_scores(agent_rl, 2)
nll_rl_bm, aic_rl_bm, bic_rl_bm = get_scores(agent_rl_bm, 2)
nll_benchmark, aic_benchmark, bic_benchmark = get_scores(agent_benchmark, 4)
nll_rnn, aic_rnn, bic_rnn = get_scores(agent_rnn, n_parameters_rnn)
nll_sindy, aic_sindy, bic_sindy = get_scores(agent_sindy, agent_sindy._count_sindy_parameters(True))

import pandas as pd

df = pd.DataFrame(
    (
        (nll_rl, aic_rl, bic_rl), 
        (nll_rl_bm, aic_rl_bm, bic_rl_bm), 
        (nll_benchmark, aic_benchmark, bic_benchmark), 
        (nll_rnn, aic_rnn, bic_rnn), 
        (nll_sindy, aic_sindy, bic_sindy),
        ),
    index = (
        'RL-Baseline', 
        'RL-Benchmark', 
        'RL-P-Benchmark', 
        'RNN', 
        'SINDy',
        ),
    columns = ('NLL', 'AIC', 'BIC'),
    )

print(df)
