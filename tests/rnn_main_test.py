import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

rnn_main.main(
    checkpoint=False,
    epochs=128,
    alpha=0.25,
    perseveration_bias=0.,
    n_submodels=1,
    bagging=False,
    analysis=True,
    hidden_size=8,
    dropout=0,
)