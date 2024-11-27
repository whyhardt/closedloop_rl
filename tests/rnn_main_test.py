import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

losses = []
# for i in range(2, 3):
_, loss = rnn_main.main(
    checkpoint=True,
    epochs_train=1024,
    # epochs_finetune=1024,
    
    model=f'params/benchmarking/rnn_sugawara_noBN.pkl',
    data = 'data/2arm/sugawara2021_143_processed.csv',
    
    hidden_size=32,
    n_actions=2,
    
    dropout=0.25,
    bagging=True,
    # weight_decay=1e-4,

    lr_train=1e-4,
    n_oversampling_train=-1,
    batch_size_train=-1,
    
    lr_finetune=1e-4,
    n_oversampling_finetune=-1,
    batch_size_finetune=-1,
    
    # n_sessions=4*1024,
    # n_trials_per_session=64,
    # sigma=0.1,
    # beta=3.,
    # alpha=0.25,
    # forget_rate=0.2,
    # alpha_penalty=0.5,
    # confirmation_bias=0.5,
    # perseverance_bias=0.25,
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