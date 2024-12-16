import sys, os

import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import rl_model
from resources.model_evaluation import plot_traces

# model = 'benchmarking/params/sugawara2021_143/hierarchical/traces_hbi_ApAcBcBr.nc'
model = 'benchmarking/params/eckstein2022_291/hierarchical/traces_hbi_ApBr.nc'

with open(model, 'rb') as file:
    mcmc = pickle.load(file)

mcmc.print_summary()

plot_traces(model, None)