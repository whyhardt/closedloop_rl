import numpy as np


def log_likelihood(data, probs):
    # data: array of binary observations (0 or 1)
    # probs: array of predicted probabilities for outcome 1
    
    # Compute log-likelihood for each data point
    log_likelihood = np.log(probs[:, data]) #+ np.log(1 - probs[:, 1 - data])
    
    # Sum over all data points and negate the result
    return np.sum(log_likelihood)


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