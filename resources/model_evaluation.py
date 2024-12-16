import sys, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from typing import Union
import numpyro

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import get_update_dynamics
from benchmarking.hierarchical_bayes_numpyro import rl_model


def log_likelihood(data, probs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1 
    
    # Sum over all data points and negate the result
    return np.sum(data * np.log(probs))


def bayesian_information_criterion(data, probs, n_parameters, ll=None):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if ll is None:
        ll = log_likelihood(data, probs)
    
    return -2 * ll + n_parameters * np.log(len(data))

def akaike_information_criterion(data, probs, n_parameters, ll=None):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    # n_parameters: integer number of trainable model parameters
    
    if ll is None:
        ll = log_likelihood(data, probs)
    
    return -2 * ll + 2 * n_parameters

def get_scores(experiment, agent, n_parameters) -> float:
        probs = get_update_dynamics(experiment, agent)[1]
        ll = log_likelihood(np.eye(2)[experiment.choices.astype(int)], probs)
        bic = bayesian_information_criterion(np.eye(2)[experiment.choices.astype(int)], probs, n_parameters, ll)
        aic = akaike_information_criterion(np.eye(2)[experiment.choices.astype(int)], probs, n_parameters, ll)
        nll = -ll
        return nll, aic, bic
    
def get_scores_array(experiment, agent, n_parameters, verbose=False) -> pd.DataFrame:
        nll, bic, aic = np.zeros((len(experiment))), np.zeros((len(experiment))), np.zeros((len(experiment)))
        ids = range(len(experiment))
        for i in tqdm(ids):
            try:
                nll_i, aic_i, bic_i = get_scores(experiment[i], agent[i], n_parameters[i])
            except ValueError:
                nll_i, aic_i, bic_i = np.nan, np.nan, np.nan
                print(f'Session {i} could not be calculated due to ValueError (most likely SINDy)')
            nll[i] += nll_i
            bic[i] += bic_i
            aic[i] += aic_i
        if verbose:
            print('Summarized statistics:')
            print(f'NLL = {np.sum(np.array(nll))} --- BIC = {np.sum(np.array(bic))} --- AIC = {np.sum(np.array(aic))}')
        return pd.DataFrame({'Job_ID': ids, 'NLL': nll, 'BIC': bic, 'AIC': aic})
    
def plot_traces(file_numpyro: Union[str, numpyro.infer.MCMC], figsize=(12, 8)):
    """
    Plot trace plots for posterior samples.

    Parameters:
    - samples: dict, where keys are parameter names and values are arrays of samples.
    - param_names: list of str, parameter names to include in the plot.
    - figsize: tuple, size of the figure.
    """
    plt.rc('font', size=7)
    plt.rc('axes', titlesize=7)
    plt.rc('axes', labelsize=7)
    plt.rc('xtick', labelsize=7)
    plt.rc('ytick', labelsize=7)
    
    if isinstance(file_numpyro, str):
        with open(file_numpyro, 'rb') as file:
            mcmc = pickle.load(file)
    elif isinstance(file_numpyro, numpyro.infer.MCMC):
        mcmc = file_numpyro
    else:
        raise AttributeError('argument 0 (file_numpyro) is not of class str or numpyro.infer.MCMC.')
    
    samples = mcmc.get_samples()
    param_names = list(mcmc.get_samples().keys())
    
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 2, figsize=figsize, gridspec_kw={'width_ratios': [2, 2]})

    for i, param in enumerate(param_names):
        param_samples = samples[param]
        
        # Trace plot
        axes[i, 1].plot(param_samples, alpha=0.7, linewidth=0.7)
        # axes[i, 1].set_title(f"Trace Plot: {param}")
        # axes[i, 1].set_ylabel(param)
        # axes[i, 1].set_xlabel("Iteration")

        # KDE plot
        sns.kdeplot(param_samples, ax=axes[i, 0], fill=True, color="skyblue", legend=False)
        # axes[i, 0].set_title(f"Posterior: {param}")
        # axes[i, 0].set_xlabel(param)
        y_label = param
        check_for = ['mean', 'std']
        for check in check_for:
            if check in y_label:
                # remove param name and keep only remaining part (mean or std)
                param = param[param.find(check):]
        axes[i, 0].set_ylabel(param)

    plt.tight_layout()
    plt.show()