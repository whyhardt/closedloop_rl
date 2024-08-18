import os

import pandas as pd
import numpy as np
import torch

from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

from rnn_main import main as rnn_main
from sindy_main import main as sindy_main
from resources.rnn_training import ensembleTypes
from resources import bandits

# script to run an entire noise analysis for different ensemble sizes (i.e. 1, 4, 8, 16, 32)
# what to extract from the different main methods:
# 1. rnn_main --> model, test loss
# 2. sindy_main --> coefficients

def main(
    iterations = 2,
    betas = [1, 3],#[1000, 5, 3, 1]
    n_submodels = [1, 2, 3],#8, 16, 32]
    epochs = 1,
    ):
    
    if not os.path.exists('noise_analysis'):
        os.makedirs('noise_analysis')
    if not os.path.exists('noise_analysis/plots'):
        os.makedirs('noise_analysis/plots')
    
    # ground truth parameters
    alpha = 0.25
    forget_rate = 0.1
    perseveration_bias = 0.25
    regret = True

    # dataset parameters
    n_trials_per_session = 50
    n_sessions = 4096
    n_actions = 2
    sigma = 0.2

    # setup features of ground truth
    features_general = ['beta', 'n_submodels', 'loss']
    feature_xQf = ['C_xQf', 'xQf']
    coeffs_xQf = [0.5*forget_rate, 1-forget_rate]
    feature_xQr_r = ['C_xQr_r', 'xQr_r', 'cdQr_r[k-1]', 'cdQr_r[k-2]']
    coeffs_xQr_r = [alpha, 1-alpha, 0, 0]
    feature_xQr_p = ['C_xQr_p', 'xQr_p', 'cdQr_p[k-1]', 'cdQr_p[k-2]']
    coeffs_xQr_p = [0, 1-alpha*2, 0, 0]
    feature_xH = ['C_xH', 'xH']
    coeffs_xH = [perseveration_bias, 1]
    features_data = ['beta', 'n_submodels', 'loss'] + ['beta_approx'] + feature_xQf + feature_xQr_r + feature_xQr_p + feature_xH
    coeffs_ground_truth = [np.array([0, 0, 0, 0] + coeffs_xQf + coeffs_xQr_r + coeffs_xQr_p + coeffs_xH)]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coeffs = []
    for b in betas:
        # setup of datasets
        environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions)
        agent = bandits.AgentQ(alpha, b, n_actions, forget_rate, perseveration_bias, False, regret)  
        print('Setup of the environment and agent complete.')

        print('Creating the training dataset...', end='\r')
        dataset_train, _ = bandits.create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=n_trials_per_session,
            n_sessions=n_sessions,
            device=device)    

        print('Creating the validation dataset...', end='\r')
        dataset_val, _ = bandits.create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=50,
            n_sessions=16,
            device=device)

        print('Creating the test dataset...', end='\r')
        dataset_test, experiment_list_test = bandits.create_dataset(
            agent=agent,
            environment=environment,
            n_trials_per_session=200,
            n_sessions=128,
            device=device)
        
        for n in n_submodels:
            for i in range(iterations):
                loss_test = rnn_main(
                    n_submodels=n,
                    epochs=epochs,
                    bagging=True,
                    ensemble=ensembleTypes.AVERAGE,
                    evolution_interval=None,
                    
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    dataset_test=dataset_test,
                    experiment_list_test=experiment_list_test,
                    
                    sigma=sigma,
                    n_actions=n_actions,
                    
                    alpha=alpha,
                    beta=b,
                    forget_rate=forget_rate,
                    perseveration_bias=perseveration_bias,
                    regret=regret,
                    )

                features, coeffs_bni = sindy_main(
                    alpha = alpha,
                    beta = b,
                    forget_rate = forget_rate,
                    perseveration_bias = perseveration_bias,
                    regret = regret,
                )

                features = ['beta', 'n_submodels', 'loss'] + list(features)
                coeffs_bni = [b, n, loss_test] + list(coeffs_bni)
                coeffs.append(np.round(np.array(coeffs_bni), 3))

    coeffs = coeffs_ground_truth + coeffs
    df = pd.DataFrame(np.stack(coeffs), columns=features_data)
    df.to_csv('noise_analysis/noise_analysis_recovered_params.csv')

    print(df)

    submodels = df['n_submodels'].unique()
    submodels = submodels[submodels != 0]
    betas = df['beta'].unique()
    betas = betas[betas != 0]
    df_without_ground_truth = df[df['n_submodels'] != 0]
    for b in betas:
        sns.boxplot(x='n_submodels', y='loss', data=df_without_ground_truth[df_without_ground_truth['beta'] == b])
        plt.savefig(f'noise_analysis/plots/noise_analysis_loss_beta_{int(b)}.png')
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
        
        
if __name__ == '__main__':
    main()
        