import os
import sys

import numpy as np
import argparse
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.model_evaluation import akaike_information_criterion, bayesian_information_criterion, log_likelihood
from utils.setup_agents import setup_agent_rnn, setup_agent_sindy, setup_custom_q_agent
from utils.convert_dataset import convert_dataset
from resources.bandits import get_update_dynamics

def main(data, model, output_file, job_id):
    
    agent_rnn, agent_sindy, agent_mcmc = None, None, None
    n_parameters_rnn, n_parameters_sindy, n_parameters_mcmc = None, None, None
    if 'rnn' in model:
        # setup rnn agent for comparison
        agent_rnn = setup_agent_rnn(model.replace('.', str(i)+'.'))
        n_parameters_rnn = sum(p.numel() for p in agent_rnn._model.parameters() if p.requires_grad)

        # setup sindy agent and get number of sindy coefficients which are not 0
        agent_sindy = setup_agent_sindy(model, data)
        n_parameters_sindy = agent_sindy._count_sindy_parameters(without_self=True)
    else:
        with open(model, 'rb') as file:
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
            if not 'mean' in p and not 'std' in p:
                parameters[p] = np.mean(mcmc.get_samples()[p][:, job_id])
                n_parameters_mcmc += 1        

        if parameters['alpha_neg'] == -1:
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

        agent_mcmc = setup_custom_q_agent(update_rule, get_q)

    # load data
    experiment = convert_dataset(data)[1][job_id]
    
    def get_scores(agent, n_parameters) -> float:
        probs = get_update_dynamics(experiment, agent)[1]
        ll_tmp = log_likelihood(experiment.choices, probs[:, 0])
        bic = bayesian_information_criterion(experiment.choices, probs[:, 0], n_parameters, ll_tmp)
        aic = akaike_information_criterion(experiment.choices, probs[:, 0], n_parameters, ll_tmp)
        ll = ll_tmp
        # nll = -ll
        return ll, aic, bic

    if agent_mcmc is not None:
        ll, aic, bic = get_scores(agent_mcmc, n_parameters_mcmc)
    if agent_rnn is not None:
        ll, aic, bic = get_scores(agent_rnn, n_parameters_rnn)
    if agent_sindy is not None:
        ll, aic, bic = get_scores(agent_sindy, n_parameters_sindy)

    # Prepare data to append
    results = pd.DataFrame({'Job_ID': [job_id], 'LL': [ll], 'BIC': [bic], 'AIC': [aic]})

    # Append results to the CSV file
    with open(output_file, 'a') as f:
        results.to_csv(f, header=f.tell()==0, index=False)  # Write header only if file is empty
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, required=True, help='SLURM job array ID')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file')
    parser.add_argument('--model', type=str, required=True, help='model file (either RNN (torch; in this case without job_id in the end) or MCMC (numpyro; all job-ids are already in the file))')
    parser.add_argument('--data', type=str, required=True, help='File with experimental data')
    args = parser.parse_args()
    
    main(args.data, args.model, args.output_file, args.job_id)