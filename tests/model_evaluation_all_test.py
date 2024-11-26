import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_evaluation_all import main
from benchmarking.hierarchical_bayes_numpyro import rl_model

# benchmarking_models = ('ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr')

# for bm in benchmarking_models:
#     data = 'data/sugawara2021_143_processed.csv'
#     job_id = np.arange(0, 143)
#     model = 'benchmarking/params/traces_'+bm+'.nc'
#     # model = 'params/benchmarking/rnn_sugawara2021_143_1_finetuned.pkl'
#     output_file = 'benchmarking/results/results_sugawara2021_143.csv'

#     main(data, model, output_file, job_id)
    
data = 'data/sugawara2021_143_processed.csv'
job_id = np.arange(0, 143)
# model = 'benchmarking/params/traces_'+bm+'.nc'
model = 'params/benchmarking/sugawara_finetuned/rnn_sugawara2021_143_6_finetuned.pkl'
output_file = 'benchmarking/results/results_sugawara2021_143.csv'

main(data, model, output_file, job_id)