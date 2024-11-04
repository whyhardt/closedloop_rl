import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sindy_main


features_list = []
for i in range(1):
    i = 1
    _, _ , features, fig, axs = sindy_main.main(
        # model = f'params/neurips2024/full_model_{i}.pkl',
        model = 'params/benchmarking/sugawara2021_143_19.pkl',
        data = 'data/sugawara2021_143_processed.csv',
        
        # sindy parameters
        polynomial_degree=2,
        threshold=0.05,
        
        # generated training dataset parameters
        n_trials_per_session = 1024,
        n_sessions = 16,
        
        alpha=0.25,
        alpha_penalty=0.5,
        confirmation_bias=0.5,
        forget_rate=0.2,
        perseveration_bias=0.25,
        beta=3.,
        
        analysis=True,
        
        verbose=True,
    )

    features_list.append(np.concatenate([np.array(features[key][1]).reshape(1, -1) for key in features], axis=-1))

features_list = np.concatenate(features_list, axis=0)
print(features_list)
mean = np.mean(features_list, axis=0).reshape(1, -1)
std = np.std(features_list, axis=0).reshape(1, -1)
features = list(np.concatenate([np.array(features[key][0]).reshape(1, -1) for key in features], axis=-1).reshape(-1))

import pandas as pd
columns = []
data = []
for i, key in enumerate(features):
    # print(f'{key}: {mean[i]:.2f} +- {std[i]:.2f}')
    if key == '1':
        key += '_'+str(i)
    columns.append(key)
    data.append(f'{mean[0, i]:.2f}+-{std[0, i]:.2f}')
df = pd.DataFrame(data=np.round(np.concatenate((mean, std)), 2), columns=columns, index=['Recovered mean', 'Recovered std'])
print(df)
print(df.to_latex(float_format="%.2f"))