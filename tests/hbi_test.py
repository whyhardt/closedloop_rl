import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main

model = 'ApAnBr'
data = 'data/sugawara2021_143_processed.csv'
main(data, model, 5, 0, 1, True)

with open(f'benchmarking/params/traces_'+model+'.nc', 'rb') as file:
        mcmc = pickle.load(file)

mcmc.print_summary()

# az.plot_trace(idata)
# plt.show()

# az.summary(idata)