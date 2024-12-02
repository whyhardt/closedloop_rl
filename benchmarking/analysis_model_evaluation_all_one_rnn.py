import os
import sys

import numpy as np
import argparse
import pickle
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import akaike_information_criterion, bayesian_information_criterion, log_likelihood, get_scores_array
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_benchmark_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model

def main(data, model):
    
    agent_rnn, agent_sindy = None, None
    n_parameters_rnn, n_parameters_sindy = None, None
    
    agent_rnn, n_parameters_rnn = [], []
    agent_sindy, n_parameters_sindy = [], []
    
    # setup rnn agent for comparison
    agent_rnn = setup_agent_rnn(model)
    n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

    # setup sindy agent and get number of sindy coefficients which are not 0
    agent_sindy = setup_agent_sindy(model, data)
    n_parameters_sindy = agent_sindy._count_sindy_parameters(without_self=True)
    
    # load data
    experiment = convert_dataset(data)[1]
    
    experiment_ids = np.arange(0, len(experiment)) 
    get_scores_array(experiment_ids, experiment, [agent_rnn]*len(experiment), [n_parameters_rnn]*len(experiment), verbose=True)
    get_scores_array(experiment_ids, experiment, [agent_sindy]*len(experiment), [n_parameters_sindy]*len(experiment), verbose=True)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model file (either RNN (torch; without job-id in the end) or MCMC (numpyro))')
    parser.add_argument('--data', type=str, required=True, help='File with experimental data')
    args = parser.parse_args()
    
    main(args.data, args.model, args.output_file, args.job_id)