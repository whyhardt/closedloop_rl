import sys
import os

import arviz as az
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main


data = 'data/data_rnn_a025_b30_p025_ap05_varMean.csv'
idata = main(data, 10000, 2500, 1, True)

az.plot_trace(idata)
plt.show()

az.summary(idata)