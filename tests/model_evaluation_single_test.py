import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_evaluation_single import main


data = 'data/sugawara2021_143_processed.csv'
job_id = 2
model = 'benchmarking/params/traces_ApBr.nc'
output_file = 'results_ApBr.csv'

main(data, model, output_file, job_id)