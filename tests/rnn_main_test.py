import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

rnn_main.main(
    checkpoint=False,
    epochs=1024,
    
    n_submodels=1,
    dropout=0.1,
    bagging=False,
    hidden_size=8,
    
    alpha=0.25,
    forget_rate=0.,
    regret=False,
    confirmation_bias=False,
    perseveration_bias=0.,
    directed_exploration_bias=0.,
    
    analysis=True,
)