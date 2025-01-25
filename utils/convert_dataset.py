import sys
import os

from typing import List

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn_utils import DictDataset
from resources.bandits import BanditSession

def convert_dataset(file: str, device = None, sequence_length: int = None, counterfactual: bool = False):
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
    reward_cols = []
    if counterfactual and 'reward_0' in df.columns:
        for action in n_actions:
            if f'reward_{action}' in df.columns:
                reward_cols.append('reward_{action}') 
    else:
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
    
    xs = {
        'c_Action': np.zeros((len(sessions), max_trials, n_actions)), 
        'c_Reward': np.zeros((len(sessions), max_trials, n_actions)) - 1,
        'c_ParticipantID': np.zeros((len(sessions), max_trials, 1)),
        }
    ys = {'y_Action': np.zeros((len(sessions), max_trials, n_actions))}
    
    probs_choice = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_action = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_reward = np.zeros((len(sessions), max_trials, n_actions)) - 1
    values_choice = np.zeros((len(sessions), max_trials, n_actions)) - 1
    
    experiment_list = []
    for index_session, s in enumerate(sessions):
        choice = np.eye(n_actions, dtype=np.float32)[df[df['session'] == s]['choice'].values.astype(int)]
        rewards = np.zeros((len(choice), n_actions), dtype=np.float32) - 1
        
        # Case 1: Single 'reward' column
        if len(reward_cols) == 1:
            reward = df[df['session'] == s][reward_cols[0]].values
            for trial in range(choice.shape[0]):
                action_idx = np.argmax(choice[trial])  # Find the index of the chosen action
                rewards[trial, action_idx] = reward[trial]
        
        # Case 2: Multiple reward columns (one for each action)
        else:
            for action_idx, c in enumerate(reward_cols):
                reward = df[df['session'] == s][c].values
                for trial in range(choice.shape[0]):
                    rewards[trial, action_idx] = reward[trial]
        
        # write arrays for DatasetRNN
        xs['c_Action'][index_session, :len(choice)] = choice
        xs['c_Reward'][index_session, :len(choice)] = rewards
        xs['c_ParticipantID'][index_session, :, 0] += index_session
        ys['y_Action'][index_session, :len(choice)-1] = choice[1:]

        experiment = BanditSession(
            choices=df[df['session'] == s]['choice'].values,
            rewards=rewards,
            session=np.full((*rewards.shape[:-1], 1), index_session),
            reward_probabilities=np.zeros_like(choice)+0.5,
            q=np.zeros_like(choice)+0.5,
            n_trials=len(choice)
        )
        
        # get update dynamics if available - only for generated data with e.g. utils/create_dataset.py
        if 'choice_prob_0' in df.columns:
            for index_action in range(n_actions):
                probs_choice[index_session, :len(choice), index_action] = df[df['session'] == s][f'choice_prob_{index_action}'].values
                values_action[index_session, :len(choice), index_action] = df[df['session'] == s][f'action_value_{index_action}'].values
                values_reward[index_session, :len(choice), index_action] = df[df['session'] == s][f'reward_value_{index_action}'].values
                values_choice[index_session, :len(choice), index_action] = df[df['session'] == s][f'choice_value_{index_action}'].values
        
        experiment_list.append(experiment)
        
    return DictDataset(xs, ys, sequence_length=sequence_length), experiment_list, df, (probs_choice, values_action, values_reward, values_choice)