import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main
    
losses = []
for i in range(1):
    loss = rnn_main.main(
        checkpoint=False,
        epochs=1,
        # model=f'params/benchmarking/sugawara2021_143_{1}.pkl',
        # data = 'data/sugawara2021_143_processed.csv',

        dropout=0.1,
        bagging=False,
        learning_rate=0.01,
        # weight_decay=1e-3,
        
        sigma=0.1,
        n_sessions=64,#4096,
        n_oversampling=-1,
        batch_size=32,
        
        alpha=0.25,
        forget_rate=0.2,
        alpha_penalty=0.5,
        confirmation_bias=0.5,
        perseveration_bias=0.25,
        beta=3.,
        
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