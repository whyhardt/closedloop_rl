import sys, os

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn_utils import parameter_file_naming
from resources.bandits import create_dataset, AgentQ, EnvironmentBanditsDrift, get_update_dynamics


n_sessions = 1024
n_trials_per_session = 256

agent = AgentQ(
    beta=3.,
    alpha=0.25,
    alpha_penalty=0.5,
    confirmation_bias=0.5,
    forget_rate=0.2, 
    perseverance_bias=0.25, 
    parameter_variance=-1,
    )

environment = EnvironmentBanditsDrift(sigma=0.1)

dataset, experiment_list, parameter_list = create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions,
            sample_parameters=True,
            )

session, choice, reward, choice_prob_0, choice_prob_1, action_value_0, action_value_1, reward_value_0, reward_value_1, choice_value_0, choice_value_1, beta, alpha, alpha_penalty, confirmation_bias, forget_rate, perseverance_bias, mean_beta, mean_alpha, mean_alpha_penalty, mean_confirmation_bias, mean_forget_rate, mean_perseverance_bias = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i, experiment in enumerate(experiment_list):
    # get update dynamics
    qs, choice_probs, _ = get_update_dynamics(experiment, agent)
    
    # append behavioral data
    session += list(np.zeros((len(experiment.choices)), dtype=int) + i)
    choice += list(experiment.choices)
    reward += list(experiment.rewards)
    
    # append update dynamics
    choice_prob_0 += list(choice_probs[:, 0])
    choice_prob_1 += list(choice_probs[:, 1])
    action_value_0 += list(qs[0][:, 0])
    action_value_1 += list(qs[0][:, 1])
    reward_value_0 += list(qs[1][:, 0])
    reward_value_1 += list(qs[1][:, 1])
    choice_value_0 += list(qs[2][:, 0])
    choice_value_1 += list(qs[2][:, 1])
    
    # append all model parameters for each trial
    beta += list(np.zeros((len(experiment.choices))) + parameter_list[i]['beta'])
    alpha += list(np.zeros((len(experiment.choices))) + parameter_list[i]['alpha'])
    alpha_penalty += list(np.zeros((len(experiment.choices))) + parameter_list[i]['alpha_penalty'])
    confirmation_bias += list(np.zeros((len(experiment.choices))) + parameter_list[i]['confirmation_bias'])
    forget_rate += list(np.zeros((len(experiment.choices))) + parameter_list[i]['forget_rate'])
    perseverance_bias += list(np.zeros((len(experiment.choices))) + parameter_list[i]['perseverance_bias'])
    
    # append all mean model parameters for each trial
    mean_beta += list(np.zeros((len(experiment.choices))) + agent._mean_beta)
    mean_alpha += list(np.zeros((len(experiment.choices))) + agent._mean_alpha)
    mean_alpha_penalty += list(np.zeros((len(experiment.choices))) + agent._mean_alpha_penalty)
    mean_confirmation_bias += list(np.zeros((len(experiment.choices))) + agent._mean_confirmation_bias)
    mean_forget_rate += list(np.zeros((len(experiment.choices))) + agent._mean_forget_rate)
    mean_perseverance_bias += list(np.zeros((len(experiment.choices))) + agent._mean_perseverance_bias)

columns = ['session', 'choice', 'reward', 'choice_prob_0', 'choice_prob_1', 'action_value_0', 'action_value_1', 'reward_value_0', 'reward_value_1', 'choice_value_0', 'choice_value_1', 'beta', 'alpha', 'alpha_penalty', 'confirmation_bias', 'forget_rate', 'perseverance_bias', 'mean_beta', 'mean_alpha', 'mean_alpha_penalty', 'mean_confirmation_bias', 'mean_forget_rate', 'mean_perseverance_bias']
data = np.array((session, choice, reward, choice_prob_0, choice_prob_1, action_value_0, action_value_1, reward_value_0, reward_value_1, choice_value_0, choice_value_1, beta, alpha, alpha_penalty, confirmation_bias, forget_rate, perseverance_bias, mean_beta, mean_alpha, mean_alpha_penalty, mean_confirmation_bias, mean_forget_rate, mean_perseverance_bias)).swapaxes(1, 0)
df = pd.DataFrame(data=data, columns=columns)

dataset_name = parameter_file_naming('data/data', np.round(agent._mean_alpha, 2), np.round(agent._mean_beta, 2), np.round(agent._mean_forget_rate, 2), np.round(agent._mean_perseverance_bias, 2), np.round(agent._mean_alpha_penalty, 2), np.round(agent._mean_confirmation_bias, 2), np.round(agent._parameter_variance, 2)).replace('.pkl','.csv')
df.to_csv(dataset_name, index=False)