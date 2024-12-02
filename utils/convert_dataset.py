import sys
import os

from typing import List

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn_utils import DatasetRNN
from resources.bandits import BanditSession

def convert_dataset(file: str, device = None) -> tuple[DatasetRNN, List[BanditSession]]:
    df = pd.read_csv(file, index_col=None)
    
    # replace all nan values with -1
    df = df.replace(np.nan, -1)
    
    # get all different sessions
    sessions = df['session'].unique()
    # get maximum number of trials per session
    max_trials = df.groupby('session').size().max()

    # let actions begin from 0
    choices = df['choice'].values
    choice_min = np.nanmin(choices[choices != -1])
    choices[choices != -1] = choices[choices != -1] - choice_min
    df['choice'] = choices
    
    # number of possible actions
    n_actions = int(df['choice'].max() + 1)
    
    # normalize rewards
    r_min = df['reward'].min()
    r_max = df['reward'].max()
    df['reward'] = (df['reward'] - r_min) / (r_max - r_min)
    
    xs = np.zeros((len(sessions), max_trials, n_actions + 2)) - 1
    ys = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    choice_probs = np.zeros((len(sessions), max_trials, n_actions)) - 1
    action_values = np.zeros((len(sessions), max_trials, n_actions)) - 1
    reward_values = np.zeros((len(sessions), max_trials, n_actions)) - 1
    choice_values = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    experiment_list = []
    for i, s in enumerate(sessions):
        choice = np.eye(n_actions)[df[df['session'] == s]['choice'].values.astype(int)]
        reward = df[df['session'] == s]['reward'].values
        session = df[df['session'] == s]['session'].values
        xs[i, :len(choice), :-2] = choice
        xs[i, :len(choice), -2] = reward
        xs[i, :len(choice), -1] = session
        ys[i, :len(choice)-1] = choice[1:]

        experiment = BanditSession(
            choices=df[df['session'] == s]['choice'].values,
            rewards=reward,
            reward_probabilities=np.zeros_like(choice)+0.5,
            q=np.zeros_like(choice)+0.5,
            n_trials=len(choice)
        )
        
        # get update dynamics if available - only for generated data with e.g. utils/create_dataset.py
        if 'choice_prob_0' in df.columns:
            choice_probs[i, :len(choice), 0] = df[df['session'] == s]['choice_prob_0'].values
            choice_probs[i, :len(choice), 1] = df[df['session'] == s]['choice_prob_1'].values
            action_values[i, :len(choice), 0] = df[df['session'] == s]['action_value_0'].values
            action_values[i, :len(choice), 1] = df[df['session'] == s]['action_value_1'].values
            reward_values[i, :len(choice), 0] = df[df['session'] == s]['reward_value_0'].values
            reward_values[i, :len(choice), 1] = df[df['session'] == s]['reward_value_1'].values
            choice_values[i, :len(choice), 0] = df[df['session'] == s]['choice_value_0'].values
            choice_values[i, :len(choice), 1] = df[df['session'] == s]['choice_value_1'].values
        
        experiment_list.append(experiment)
        
    return DatasetRNN(xs, ys, device=device), experiment_list, df, (choice_probs, action_values, reward_values, choice_values)