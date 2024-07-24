import pandas as pd
import numpy as np
from scipy.stats import norm

df = pd.read_csv('params.csv')
print(df)

# get mean and std grouped by beta and n_submodels
grouped = df.groupby(['beta', 'n_submodels'])
group_stats = grouped.aggregate(['mean', 'std']).reset_index()

print(group_stats)

import seaborn as sns
import matplotlib.pyplot as plt

# make matrix of plots for each beta
# the columns are the n_submodels
# the rows are the coefficients given by the columns of the dataframe
# each plot shows on the x-axis the real value given in the group (0, 0) and a normal distribution given by the mean and std of one n_submodel

# Plotting
real_values = df[df['beta'] == 0]  # Extracting real values from group (0, 0)
coefficients = real_values.columns[3:]  # Extracting coefficients
submodels = group_stats['n_submodels'].unique()
submodels = submodels[submodels != 0]
betas = df['beta'].unique()
betas = betas[betas != 0]

for b in betas:
    fig, axes = plt.subplots(nrows=len(coefficients), ncols=len(submodels), figsize=(15, 10), constrained_layout=True)
    for i, coeff in enumerate(coefficients):
        real_value = real_values[coeff].values
        for j, submodel in enumerate(submodels):
            mean_val = np.round(group_stats[(group_stats['beta'] == b) & (group_stats['n_submodels'] == submodel)][(coeff, 'mean')].values[0], 3)
            std_val = np.round(group_stats[(group_stats['beta'] == b) & (group_stats['n_submodels'] == submodel)][(coeff, 'std')].values[0], 3)
            
            ax = axes[i, j]
            # sns.histplot(real_value, kde=False, ax=ax, color='blue', stat='density', label='Real Values')
            # plot real value as a vertical dashed line
            ax.axvline(x=real_value, color='blue', linestyle='--', label='Real Value')
            
            # Plot normal distribution
            x = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, 100)
            p = norm.pdf(x, mean_val, std_val)
            ax.plot(x, p, 'k', linewidth=2, label=f'N({mean_val:.2f}, {std_val:.2f})')
            
            ax.set_xlim(real_value-1, real_value+1)
            # turn off x labels for all but the last row
            if i != len(coefficients) - 1:
                ax.set_xticklabels([])
            else:
                # set x labels to the real value and real value +- 1
                # ax.set_xticks([real_value-1, real_value, real_value+1])
                pass
            ax.set_yticklabels([])
            
            # ax.set_title(f'{coeff} - Submodel {submodel}')
            # ax.set_xlabel('Value')
            if i == 0:
                ax.set_title(f'n_submodels {int(submodel)}')
            if j == 0:
                ax.set_ylabel(coeff)
            # ax.legend()

    plt.savefig(f'noise_analysis/plots/noise_analysis_coeffs_beta_{b}.png')
    plt.close(fig)
    