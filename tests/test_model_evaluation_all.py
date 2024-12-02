import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_evaluation_all import main
from benchmarking.hierarchical_bayes_numpyro import rl_model

benchmarking_models = ('ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr', 'ApAnBcBr', 'ApAnAcBcBr')

for bm in benchmarking_models:
    data = 'data/2arm/eckstein2022_291_processed.csv'
    job_id = np.arange(0, 143)
    model = 'benchmarking/params/eckstein2022_291/hierarchical/traces_hbi_'+bm+'.nc'
    # model = 'params/benchmarking/rnn_sugawara2021_143_1_finetuned.pkl'
    output_file = 'benchmarking/results/results_eckstein.csv'

    main(data, model, output_file, job_id)

# bm = 'ApAnBcBr'
# data = 'data/2arm/sugawara2021_143_processed.csv'
# job_id = np.arange(0, 142)
# model = 'benchmarking/params/sugawara2021_143/hierarchical/traces_'+bm+'.nc'
# output_file = 'benchmarking/results/results_eckstein2022_291.csv'

# main(data, model, output_file, job_id)