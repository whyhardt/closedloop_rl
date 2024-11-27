import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_evaluation_all_one_rnn import main
from benchmarking.hierarchical_bayes_numpyro import rl_model

for i in range(6,7):
    data = 'data/2arm/sugawara2021_143_processed.csv'
    job_id = np.arange(0, 143)
    # model = 'benchmarking/params/traces_'+bm+'.nc'
    model = f'params/benchmarking/rnn_sugawara_noBN.pkl'  # best performing model -> 6
    output_file = 'benchmarking/results/results_sugawara2021_143.csv'

    main(data, model, output_file, job_id)