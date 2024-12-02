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

def main(data, model, output_file, job_id):
    
    if isinstance(job_id, int):
        job_id = np.arange(0, job_id)
        
    agent_rnn, agent_sindy, agent_mcmc = None, None, None
    n_parameters_rnn, n_parameters_sindy, n_parameters_mcmc = None, None, None
    if 'rnn' in model:
        agent_rnn, n_parameters_rnn = [], []
        agent_sindy, n_parameters_sindy = [], []
        for i in job_id:
            # setup rnn agent for comparison
            agent_rnn.append(setup_agent_rnn(model.replace('.', str(i)+'.')))
            n_parameters_rnn.append(sum(p.numel() for p in agent_rnn[-1]._model.parameters() if p.requires_grad))

            # # setup sindy agent and get number of sindy coefficients which are not 0
            # agent_sindy.append(setup_agent_sindy(model.replace('.', str(i)+'.'), data))
            # n_parameters_sindy.append(agent_sindy[-1]._count_sindy_parameters(without_self=True))
    else:
        with open(model, 'rb') as file:
            mcmc = pickle.load(file)
        parameters = {
            'alpha_pos': np.full((len(job_id)), 1),
            'alpha_neg': np.full((len(job_id)), -1),
            'alpha_c': np.full((len(job_id)), 1),
            'beta_c': np.full((len(job_id)), 0),
            'beta_r': np.full((len(job_id)), 1),
        }
        n_parameters_mcmc = 0
        # mcmc.print_summary()
        for p in mcmc.get_samples():
            if not 'mean' in p and not 'std' in p:
                parameters[p] = np.mean(mcmc.get_samples()[p], axis=0)
                n_parameters_mcmc += 1
        n_parameters_mcmc = np.full((len(job_id)), n_parameters_mcmc)

        if np.mean(parameters['alpha_neg']) == -1:
            parameters['alpha_neg'] = parameters['alpha_pos']
        
        agent_mcmc = []
        for i in job_id:
            parameters_i = {}
            for key in parameters:
                parameters_i[key] = parameters[key][i]
            agent_mcmc.append(setup_benchmark_q_agent(parameters_i))

    # load data
    experiment = convert_dataset(data)[1]
    
    if agent_mcmc is not None:
        df = get_scores_array(job_id, experiment, agent_mcmc, n_parameters_mcmc, verbose=True)
        output_file = output_file.replace('.', '_mcmc_'+model.split('/')[-1].split('_')[-1].split('.')[0]+'.')
        df.to_csv(output_file)
        print(f'Saved MCMC scores in {output_file}')
    if agent_rnn is not None:
        df = get_scores_array(job_id, experiment, agent_rnn, n_parameters_rnn, verbose=True)
        df.to_csv(output_file.replace('.', '_rnn.'))
    # if agent_sindy is not None:
    #     df = get_scores_array(experiment, agent_sindy, n_parameters_sindy)
    #     df.to_csv(output_file.replace('.', '_sindy.'))

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, required=True, help='SLURM job array ID')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file')
    parser.add_argument('--model', type=str, required=True, help='model file (either RNN (torch; without job-id in the end) or MCMC (numpyro))')
    parser.add_argument('--data', type=str, required=True, help='File with experimental data')
    args = parser.parse_args()
    
    main(args.data, args.model, args.output_file, args.job_id)