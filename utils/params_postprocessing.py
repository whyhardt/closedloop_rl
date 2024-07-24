import pandas as pd
import numpy as np


def mean_and_std(data, split_column='n_submodels', split_value=0):
    """this function computes the mean and standard deviation of the input data columns

    data may look like this:
        n_submodels   loss   beta  C_xQf    xQf  C_xQr_r  xQr_r  cdQr_r[k-1]  cdQr_r[k-2]  C_xQr_p  xQr_p  cdQr_p[k-1]  cdQr_p[k-2]   C_xH     xH
    0           0.0  0.000  5.000  0.050  0.900    0.250  0.750        0.000          0.0    0.000  0.500        0.000          0.0  0.250  1.000
    1           1.0  0.396  2.923  0.038  0.000    0.270  0.664        0.124          0.0    0.000  0.678        0.050          0.0  0.122  1.000
    2           8.0  0.401  3.593  0.114  0.450    0.346  0.555        0.050          0.0    0.000  0.640        0.049          0.0  0.180  0.998
    3           0.0  0.000  3.000  0.050  0.900    0.250  0.750        0.000          0.0    0.000  0.500        0.000          0.0  0.250  1.000
    4           1.0  0.396  2.923  0.038  0.000    0.270  0.664        0.124          0.0    0.000  0.678        0.050          0.0  0.122  1.000
    5           8.0  0.401  3.593  0.114  0.450    0.346  0.555        0.050          0.0    0.000  0.640        0.049          0.0  0.180  0.998
    
    This function will split the data into batches according to split_column and split_value i.e. whenever the value in the split_column is equal to split_value, a new batch is created.
    
    batch #1 - name after the beta value e.g. beta_5:
        n_submodels   loss   beta  C_xQf    xQf  C_xQr_r  xQr_r  cdQr_r[k-1]  cdQr_r[k-2]  C_xQr_p  xQr_p  cdQr_p[k-1]  cdQr_p[k-2]   C_xH     xH
    0           1.0  0.396  2.923  0.038  0.000    0.270  0.664        0.124          0.0    0.000  0.678        0.050          0.0  0.122  1.000
    1           8.0  0.401  3.593  0.114  0.450    0.346  0.555        0.050          0.0    0.000  0.640        0.049          0.0  0.180  0.998
    
    batch #1 - name after the beta value e.g. beta_3:
    0           1.0  0.396  2.923  0.038  0.000    0.270  0.664        0.124          0.0    0.000  0.678        0.050          0.0  0.122  1.000
    1           8.0  0.401  3.593  0.114  0.450    0.346  0.555        0.050          0.0    0.000  0.640        0.049          0.0  0.180  0.998
    
    Get all unique values for split_column and split the batches further by these values.
    
    Then compute the mean and standard deviation of the final split for each column and return them as a dictionary of dataframes.
    The keys of the dictionary are the batch names e.g. beta_5, beta_3, etc.
    
    Args:
        data (_type_): _description_
    """
    
    if isinstance(data, str):
        data = pd.read_csv(data)
    
    # find all index values where the split_column has the split_value and split the data into batches
    split_indices = data.index[data[split_column] == split_value].tolist()
    batches = {}
    for i, idx in enumerate(split_indices[:-1]):
        beta = data.loc[idx]['beta']
        batches[beta] = data.loc[idx+1:split_indices[i+1]].copy()
    beta = data.loc[split_indices[-1]]['beta']
    batches[beta] = data.loc[split_indices[-1]+1:].copy()
    
    # split the batches further by unique values of the split_column
    unique_values = data[split_column].unique()
    # remove the split_value from the unique values
    unique_values = unique_values[unique_values != split_value]
    for name, batch in batches.items():
        split_batches = {}
        for value in unique_values:
            split_batches[value] = batch[batch[split_column] == value]
        batches[name] = split_batches
    
    # compute mean and standard deviation of each batch
    mean_std = {}
    for name, batch in batches.items():
        mean_std[name] = batch.mean(), batch.std()
        
    return mean_std
