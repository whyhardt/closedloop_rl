import numpyro
import arviz as az
from benchmarking import benchmarking_eckstein2022
import pickle
import matplotlib.pyplot as plt


# Run example
path_model_fit = 'benchmarking/params/mcmc_eckstein2022_sim_baseline_fit_benchmark.nc'
# path_model_sim = 'params/eckstein2022/mcmc_eckstein2022_benchmark.nc'
# path_model_fit = 'params/eckstein2022/mcmc_eckstein2022_sim_benchmark_fit_benchmark.nc'

rl_model = benchmarking_eckstein2022.rl_model

print("Running example MCMC analysis...")
# with open(path_model_sim, 'rb') as file:
#     mcmc_sim = pickle.load(file)

with open(path_model_fit, 'rb') as file:
    mcmc_fit = pickle.load(file)

# Get samples
# samples_sim = mcmc_sim.get_samples()
samples_fit = mcmc_fit.get_samples()

# sum_sim = az.summary(az.from_numpyro(mcmc_sim, log_likelihood=False))
sum_fit = az.summary(az.from_numpyro(mcmc_fit, log_likelihood=False))

print('Mean:')
# print('Sim:')
# print(samples_sim['alpha_pos_mean'].mean())
# print(samples_sim['alpha_neg_mean'].mean())
# print(samples_sim['beta_r_mean'].mean())
# print(samples_sim['beta_ch_mean'].mean())
print('Fit:')
print('Mean:')
print(samples_fit['alpha_pos_mean'].mean())
print(samples_fit['alpha_neg_mean'].mean())
print(samples_fit['beta_r_mean'].mean())
print(samples_fit['beta_ch_mean'].mean())
print('Std:')
print(samples_fit['alpha_pos_mean'].std())
print(samples_fit['alpha_neg_mean'].std())
print(samples_fit['beta_r_mean'].std())
print(samples_fit['beta_ch_mean'].std())

print('Individuals:')
# print('Sim:')
# print(samples_sim['alpha_pos'].mean(axis=0)[0], samples_sim['alpha_pos'].mean(axis=0)[1], samples_sim['alpha_pos'].mean(axis=0)[2])
# print(samples_sim['alpha_neg'].mean(axis=0)[0], samples_sim['alpha_neg'].mean(axis=0)[1], samples_sim['alpha_neg'].mean(axis=0)[2])
# print(samples_sim['beta_r'].mean(axis=0)[0], samples_sim['beta_r'].mean(axis=0)[1], samples_sim['beta_r'].mean(axis=0)[2])
# print(samples_sim['beta_ch'].mean(axis=0)[0], samples_sim['beta_ch'].mean(axis=0)[1], samples_sim['beta_ch'].mean(axis=0)[2])
print('Fit:')
print(samples_fit['alpha_pos'].mean(axis=0)[0], samples_fit['alpha_pos'].mean(axis=0)[1], samples_fit['alpha_pos'].mean(axis=0)[2])
print(samples_fit['alpha_neg'].mean(axis=0)[0], samples_fit['alpha_neg'].mean(axis=0)[1], samples_fit['alpha_neg'].mean(axis=0)[2])
print(samples_fit['beta_r'].mean(axis=0)[0], samples_fit['beta_r'].mean(axis=0)[1], samples_fit['beta_r'].mean(axis=0)[2])
print(samples_fit['beta_ch'].mean(axis=0)[0], samples_fit['beta_ch'].mean(axis=0)[1], samples_fit['beta_ch'].mean(axis=0)[2])
