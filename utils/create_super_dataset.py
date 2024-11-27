import pandas as pd
import numpy as np
import os

from convert_dataset import convert_dataset


data_dir = 'data/2arm/'
files = os.listdir(data_dir)

session = np.ndarray((0,))
choice = np.ndarray((0,))
reward = np.ndarray((0,))
for i, f in enumerate(files):
    if 'processed' in f:
        _, _, df, _ = convert_dataset(data_dir+f)
        # df = pd.read_csv(data_dir+f)
        session_f = df['session'].values + 10000*i
        session = np.concatenate((session, session_f))
        choice = np.concatenate((choice, df['choice'].values))
        reward = np.concatenate((reward, df['reward'].values))

pd.DataFrame(np.stack((session, choice, reward)).T, columns=('session', 'choice', 'reward')).to_csv(data_dir+'super_dataset.csv', index=False)