import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

rnn_main.main(
    checkpoint=True,
    epochs=0,
    alpha=0.25,
    perseveration_bias=0.25,
    directed_exploration_bias=0.,
    forget_rate=0.1,
    regret=True,
    confirmation_bias=True,
    n_submodels=1,
    bagging=False,
    analysis=True,
    hidden_size=8,
    dropout=0,
)