import pandas as pd
import numpy as np

from rnn_main import main as rnn_main
from sindy_main import main as sindy_main
from resources.rnn_training import ensembleTypes

# script to run an entire noise analysis for different ensemble sizes (i.e. 1, 4, 8, 16, 32)
# what to extract from the different main methods:
# 1. rnn_main --> model, test loss
# 2. sindy_main --> coefficients

# ground truth parameters
alpha = 0.25
forget_rate = 0.1
perseveration_bias = 0.25
regret = True

iterations = 5
betas = [1]#[1000, 5, 3, 1]
n_submodels = [1, 8]#8, 16, 32]
epochs = 1024

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
features_ground_truth = feature_xQf + feature_xQr_r + feature_xQr_p + feature_xH
coeffs_ground_truth = [np.array([0, 0, 0] + coeffs_xQf + coeffs_xQr_r + coeffs_xQr_p + coeffs_xH)]

coeffs = []
for b in betas:
    coeffs_b = []
    for n in n_submodels:
        for i in range(iterations):
            loss_test = rnn_main(
                n_submodels=n,
                epochs=epochs,
                sampling_replacement=True,
                ensemble=ensembleTypes.AVERAGE,
                evolution_interval=None,
                
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

            features = ['n_submodels', 'loss'] + list(features)
            coeffs_bni = [n, loss_test] + list(coeffs_bni)
            
            coeffs_b.append(np.round(np.array(coeffs_bni), 3))
    coeffs_ground_truth[0][2] = b
    coeffs_b = coeffs_ground_truth + coeffs_b
    coeffs += coeffs_b

df = pd.DataFrame(np.stack(coeffs), columns=['n_submodels', 'loss', 'beta'] + features_ground_truth)
df.to_csv('params.csv')
print(df)