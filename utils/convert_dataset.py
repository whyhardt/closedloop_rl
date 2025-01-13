import sys
import os

from typing import List

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn_utils import DatasetRNN
from resources.bandits import BanditSession

def convert_dataset(file: str, device = None, sequence_length: int = None):
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
    
    # get all columns with rewards
    # reward_cols = []
    # for c in df.columns:
    #     if 'reward' in c:
    #         reward_cols.append(c)
    reward_cols = ['reward']
    
    # normalize rewards
    r_min, r_max = [], []
    for c in reward_cols:
        r_min.append(df[c].min())
        r_max.append(df[c].max())
    r_min = np.min(r_min)
    r_max = np.max(r_max)
    for c in reward_cols:
        df[c] = (df[c] - r_min) / (r_max - r_min)
    
    xs = np.zeros((len(sessions), max_trials, n_actions*2 + 1)) - 1
    ys = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    probs_choice = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_action = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_reward = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_choice = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    experiment_list = []
    for i, s in enumerate(sessions):
        choice = np.eye(n_actions)[df[df['session'] == s]['choice'].values.astype(int)]
        rewards = np.zeros((len(choice), n_actions)) - 1
        for j, c in enumerate(reward_cols):
            reward = df[df['session'] == s][c].values
            rewards[:, j] = reward
        
        # write arrays for DatasetRNN
        xs[i, :len(choice), :n_actions] = choice
        xs[i, :len(choice), n_actions:n_actions*2] = rewards
        xs[i, :, -1] += i+1
        ys[i, :len(choice)-1] = choice[1:]

        experiment = BanditSession(
            choices=df[df['session'] == s]['choice'].values,
            rewards=rewards,
            session=np.full((*rewards.shape[:-1], 1), i),
            reward_probabilities=np.zeros_like(choice)+0.5,
            q=np.zeros_like(choice)+0.5,
            n_trials=len(choice)
        )
        
        # get update dynamics if available - only for generated data with e.g. utils/create_dataset.py
        if 'choice_prob_0' in df.columns:
            for i in range(n_actions):
                probs_choice[i, :len(choice), i] = df[df['session'] == s][f'choice_prob_{i}'].values
                values_action[i, :len(choice), i] = df[df['session'] == s][f'action_value_{i}'].values
                values_reward[i, :len(choice), i] = df[df['session'] == s][f'reward_value_{i}'].values
                values_choice[i, :len(choice), i] = df[df['session'] == s][f'choice_value_{i}'].values
        
        experiment_list.append(experiment)
        
    return DatasetRNN(xs, ys, device=device, sequence_length=sequence_length), experiment_list, df, (probs_choice, values_action, values_reward, values_choice)