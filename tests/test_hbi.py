import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

model = 'ApAnBcBr'
data = 'data/2arm/eckstein2022_291_processed.csv'
output_file = f'benchmarking/params/eckstein2022_291/non_hierarchical/traces_{model}.nc'
mcmc = main(data, model, 4096, 2048, 3, False, output_file)

# with open(output_file.split('.')[0]+model+'.nc', 'rb') as file:
#         mcmc = pickle.load(file)

mcmc.print_summary()