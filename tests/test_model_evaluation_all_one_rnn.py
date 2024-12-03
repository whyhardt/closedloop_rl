import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_evaluation_all_one_rnn import main
from benchmarking.hierarchical_bayes_numpyro import rl_model


# data = 'data/2arm/eckstein2022_291_processed.csv'
# model = f'params/benchmarking/rnn_eckstein.pkl'
data = 'data/2arm/sugawara2021_143_processed.csv'
model = f'params/benchmarking/rnn_sugawara.pkl'  # best performing model -> 6
    
main(data, model)