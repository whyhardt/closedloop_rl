import sys
import os

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


original_data = 'data/data_rnn_a05_b30_p025_ap05_varDict.csv'
inference_data = 'benchmarking/params/data_rnn_a05_b30_p025_ap05_varDict_1000s_3c.nc'

idata = az.from_netcdf(inference_data)
print(az.summary(idata))
az.plot_trace(idata)
plt.show()

# load original data
odata = pd.read_csv(original_data)
odata['perseverance_bias'] = odata['perseverance_bias'] * odata['beta']
# get true participant parameters
# keys in data for true participant parameters
okeys = ['beta', 'alpha', 'alpha_penalty', 'perseverance_bias', 'alpha_perseverance']
# standard values (if not available in model)
oparams_standard = {
    'alpha_perseverance': 1,
}
n_sessions = int(odata['session'].max())
oparams = np.zeros((n_sessions, len(okeys)))
for s in range(n_sessions):
    for i, k in enumerate(okeys):
        if k in odata.columns:
            oparams[s, i] = odata[odata['session'] == s].iloc[0][k]
        elif k in oparams_standard:
            oparams[s, i] = oparams_standard[k]

# load inferred data
idata = az.from_netcdf(inference_data)
summary = az.summary(idata)
# get inferred participant parameters
# keys in data for true participant parameters
ikeys = ['beta_r', 'alpha_pos', 'alpha_neg', 'beta_c', 'alpha_c']
iparams = np.zeros((n_sessions, len(ikeys)))
for s in range(n_sessions):
    for i, k in enumerate(ikeys):
        # oparams[s, i] = idata.posterior[k][0, :, s].mean().values
        iparams[s, i] = summary['mean'][k+f'[{s}]']

# plot histogram for each parameter
hist_oparams = []
hist_iparams = []
ranges = ((-0.2, 10.2), (-0.2, 1.2), (-0.2, 1.2), (-0.2, 1.2), (-0.2, 1.2))
n_bins = 50
fig, axs = plt.subplots(len(ikeys))
for i in range(len(ikeys)):
    # Plot the histograms
    axs[i].hist(oparams[:, i], bins=n_bins, range=ranges[i], alpha=0.5, label='oparams', color='blue')
    axs[i].hist(iparams[:, i], bins=n_bins, range=ranges[i], alpha=0.5, label='iparams', color='orange')
    
    # Add labels, legend, and title
    axs[i].set_title(f"Parameter {ikeys[i]}")
    axs[i].set_ylabel("Frequency")
axs[i].set_xlabel("Value")
axs[i].legend()

# Adjust layout for readability
# plt.tight_layout()
plt.show()
