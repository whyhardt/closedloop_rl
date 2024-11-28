import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_hierarchical_evaluation_all_one_rnn import main
from benchmarking.hierarchical_bayes_numpyro import rl_model

for i in range(6,7):
    data = 'data/2arm/eckstein2022_291_processed.csv'
    job_id = np.arange(0, 291)
    # model = 'benchmarking/params/traces_'+bm+'.nc'
    model = f'params/benchmarking/gru_eckstein.pkl'  # best performing model -> 6
    output_file = 'benchmarking/results/results_eckstein.csv'

    main(data, model, output_file, job_id)