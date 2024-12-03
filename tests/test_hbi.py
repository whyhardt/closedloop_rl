import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import jax

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main
from resources.model_evaluation import plot_traces

model = 'ApAnBcBr'
data = 'data/2arm/sugawara2021_143_processed.csv'
output_file = f'benchmarking/params/sugawara2021_143/traces_test.nc'
# with jax.disable_jit():
# with jax.default_device(jax.devices('cpu')[0]):
mcmc = main(data, model, 256, 256, 1, False, output_file)

mcmc.print_summary()
plot_traces(mcmc)