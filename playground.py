import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


df = pd.read_csv('noise_analysis/noise_analysis_recovered_params.csv')

print(df)

submodels = df['n_submodels'].unique()
submodels = submodels[submodels != 0]
betas = df['beta'].unique()
betas = betas[betas != 0]
df_without_ground_truth = df[df['n_submodels'] != 0]
for i, b in enumerate(betas):
    sns.boxplot(x='n_submodels', y='loss', data=df_without_ground_truth[df_without_ground_truth['beta'] == b])
plt.ylim(0.25, 1.0)
# plt.ylabel('Loss')
plt.savefig(f'noise_analysis/plots/noise_analysis_loss.png')
plt.close()

# get mean and std grouped by beta and n_submodels
grouped = df.groupby(['beta', 'n_submodels'])
group_stats = grouped.aggregate(['mean', 'std']).reset_index()
group_stats.to_csv('noise_analysis/noise_analysis_group_stats.csv')

print(group_stats)

# Plot matrix of plots for each beta
# the columns are the n_submodels
# the rows are the coefficients given by the columns of the dataframe
# each plot shows on the x-axis the real value given in the group (0, 0) and a normal distribution given by the mean and std of the respective n_submodels model
real_values = df[df['beta'] == 0]  # Extracting real values from group (0, 0)
coefficients = real_values.columns[3:]  # Extracting coefficients

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
            ax.set_yticklabels([])
            
            # ax.set_title(f'{coeff} - Submodel {submodel}')
            # ax.set_xlabel('Value')
            if i == 0:
                ax.set_title(f'n_submodels {int(submodel)}')
            if j == 0:
                ax.set_ylabel(coeff)

    plt.savefig(f'noise_analysis/plots/noise_analysis_coeffs_beta_{int(b)}.png')
    plt.close(fig)