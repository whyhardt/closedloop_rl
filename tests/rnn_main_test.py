import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

losses = []
for i in range(1):
    loss = rnn_main.main(
        checkpoint=False,
        epochs=128,
        model=f'params/benchmarking/sugawara2021_143_{20}.pkl',
        
        n_submodels=1,
        dropout=0.1,
        bagging=True,
        learning_rate=0.001,
        # weight_decay=0.001,
        
        data = 'data/sugawara2021_143_processed.csv',
        
        # sigma=0.1,
        # n_sessions=4096,
        # n_oversampling=-1,
        # batch_size=-1,
        
        # alpha=0.25,
        # forget_rate=0.2,
        # regret=True,
        # confirmation_bias=True,
        # perseveration_bias=0.25,
        
        analysis=True,
    )
    
    losses.append(0+loss)
    
print(losses)
import numpy as np
print('Mean: ' + str(np.mean(losses)))
print('Std: ' + str(np.std(losses)))