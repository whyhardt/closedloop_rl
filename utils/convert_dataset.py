import sys
import os

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn_utils import DatasetRNN
from resources.bandits import BanditSession

def to_datasetrnn(file: str, device = None) -> DatasetRNN:
    df = pd.read_csv(file, index_col=None)
    sessions = df['session'].unique()
    xs = np.zeros((len(sessions), 200, 3)) - 1
    ys = np.zeros((len(sessions), 200, 2)) - 1
    # lens = []
    experiment_list = []
    for i, s in enumerate(sessions):
        choice = np.eye(2)[df[df['session'] == s]['choice'].values]
        reward = df[df['session'] == s]['reward'].values
        xs[i, :len(choice), :-1] = choice
        xs[i, :len(choice), -1] = reward
        ys[i, :len(choice)-1] = choice[1:]
        
        # lens.append(len(choice))
        
        experiment = BanditSession(
            choices=df[df['session'] == s]['choice'].values,
            rewards=reward,
            reward_probabilities=np.zeros_like(choice)+0.5,
            q=np.zeros_like(choice)+0.5,
            n_trials=len(choice)
        )
        
        experiment_list.append(experiment)
        
    return DatasetRNN(xs, ys, device=device), experiment_list