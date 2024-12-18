import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

losses = []
# for i in range(2, 3):
_, loss = rnn_main.main(
    checkpoint=False,
    epochs_train=1024,
    
    # data='data/data_rnn_a025_b30_f02_p025_ap05_cb05_varMean.csv',
    # model='params/params_rnn_a025_b30_f02_p025_ap05_cb05_varMean_1.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    # model = f'params/benchmarking/rnn_sugawara.pkl',
    # data = 'data/2arm/sugawara2021_143_processed.csv',
    
    n_actions=2,
    
    dropout=0.1,
    bagging=True,
    # weight_decay=1e-4,

    lr_train=1e-2,
    n_oversampling_train=-1,
    batch_size_train=-1,
    
    train_test_ratio=0,
    n_sessions=1024,
    n_trials_per_session=256,
    sigma=0.05,
    counterfactual=False,
    # beta_reward=3.,
    # alpha=0.5,
    # alpha_counterfactual=0.5,
    # forget_rate=0.2,
    # alpha_penalty=0.5,
    # confirmation_bias=0.5,
    # beta_choice=3.0,
    # alpha_choice=0.25,
    # parameter_variance=0.,
    
    analysis=True,
)

losses.append(0+loss)
    
print(losses)
import numpy as np
mean = np.mean(losses)
mean_str = str(np.round(mean, 4)).replace('.','')
std = np.std(losses)
std_str = str(np.round(std, 4)).replace('.','')
print('Mean: ' + str(mean))
print('Std: ' + str(std))