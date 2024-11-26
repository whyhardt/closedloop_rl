import pandas as pd 
import numpy as np

benchmarking_models = ['ApBr', 'ApAnBr', 'ApBcBr', 'ApAcBcBr','ApAnBcBr', 'ApAnAcBcBr']
benchmark_cols = ('LL', 'BIC', 'AIC')

benchmarks = np.zeros((len(benchmarking_models)+1, 3))

for i, bm in enumerate(benchmarking_models):
    data_mcmc = 'benchmarking/results/results_sugawara2021_143_mcmc_'+bm+'.csv'
    df_mcmc = pd.read_csv(data_mcmc)
    benchmarks[i, 0] = df_mcmc['LL'].sum()
    benchmarks[i, 1] = df_mcmc['BIC'].sum()
    benchmarks[i, 2] = df_mcmc['AIC'].sum()
    # print(f'LL_MCMC_{bm} = {ll_mcmc}')
    
data_rnn = 'benchmarking/results/results_sugawara2021_143_rnn.csv'
df_rnn = pd.read_csv(data_rnn)
benchmarks[-1, 0] = df_rnn['LL'].sum()
benchmarks[-1, 1] = df_rnn['BIC'].sum()
benchmarks[-1, 2] = df_rnn['AIC'].sum()
# print(f'LL_RNN = {ll_rnn}')

df_bm = pd.DataFrame(benchmarks, columns=benchmark_cols, index=benchmarking_models+['RNN'])
print(df_bm)