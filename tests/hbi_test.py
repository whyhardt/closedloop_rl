import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle
import jax
import numpyro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

# models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAnBcBr', 'ApAnAcBcBr']  # 'ApAcBcBr'
models = ['ApAnBcBr']

for model in models:
        output_file = f'benchmarking/params/eckstein2022_291/traces_'+model+'.nc'
        data = 'data/2arm/eckstein2022_291_processed.csv'
        with jax.default_device(jax.devices("cpu")[0]):
                numpyro.set_platform('cpu')
                numpyro.set_host_device_count(3)
                main(data, model, 4096, 2048, 3, False, output_file)

        with open(output_file, 'rb') as file:
                mcmc = pickle.load(file)

        mcmc.print_summary()

# az.plot_trace(idata)
# plt.show()

# az.summary(idata)