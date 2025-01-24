import sys
import os

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sindy_main


features_list = []
for i in range(1):
    # i = 1
    _, _ , features = sindy_main.main(
        
        # data='data/rldm2025/data_rldm_512p_7.csv',
        # model='params/rldm2025/params_rldm_512p_7.pkl',
        
        model = 'params/benchmarking/rnn_sugawara.pkl',
        data = 'data/sugawara2021_143_processed.csv',
        
        # model = 'params/benchmarking/bahrami2020_965_0.pkl',
        # data = 'data/bahrami2020_965_processed.csv',
        
        # general recovery parameters
        session_id=142,
        n_sessions=512,
        n_trials_per_session=256,
        
        # sindy parameters
        polynomial_degree=2,
        regularization=1e-1,
        threshold=0.05,
        verbose=True,
        
        # generated training dataset parameters
        # n_actions=2,
        # sigma=0.2,
        # beta_reward=3.,
        # alpha=0.25,
        # alpha_penalty=0.5,
        # beta_choice=3.,
        # alpha_choice=1.,
        # forget_rate=0.2,
        # confirmation_bias=0.5,
        # counterfactual=True,
        # alpha_counterfactual=0.5,
        # parameter_variance=0.,
        
        analysis=True,
    )

    # features_list.append(np.concatenate([np.array(features[key][1]).reshape(1, -1) for key in features], axis=-1))

# features_list = np.concatenate(features_list, axis=0)
# print(features_list)

# mean = np.mean(features_list, axis=0).reshape(1, -1)
# std = np.std(features_list, axis=0).reshape(1, -1)
# features = list(np.concatenate([np.array(features[key][0]).reshape(1, -1) for key in features], axis=-1).reshape(-1))

# import pandas as pd
# columns = []
# data = []
# for i, key in enumerate(features):
#     # print(f'{key}: {mean[i]:.2f} +- {std[i]:.2f}')
#     if key == '1':
#         key += '_'+str(i)
#     columns.append(key)
#     data.append(f'{mean[0, i]:.2f}+-{std[0, i]:.2f}')
# df = pd.DataFrame(data=np.round(np.concatenate((mean, std)), 2), columns=columns, index=['Recovered mean', 'Recovered std'])
# print(df)
# print(df.to_latex(float_format="%.2f"))