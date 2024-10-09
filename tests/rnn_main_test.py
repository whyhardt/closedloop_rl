import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

rnn_main.main(
    perseveration_bias=0.25,
    bagging=True,
    analysis=True,
)