import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
import argparse
    

def rl_model(choice: jnp.array, reward: jnp.array):
    
    # Priors for group-level parameters
    alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Uniform(low=0.01, high=0.99))
    alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Uniform(low=0.01, high=0.99))
    pers_mean = numpyro.sample("pers_mean", dist.Uniform(low=0.01, high=0.99))
    beta_mean = numpyro.sample("beta_mean", dist.Uniform(low=0.01, high=9.99))
    
    # Priors for individual-level variation (hierarchical)
    alpha_pos_std = numpyro.sample("alpha_pos_std", dist.HalfNormal(0.3))
    alpha_neg_std = numpyro.sample("alpha_neg_std", dist.HalfNormal(0.3))
    pers_std = numpyro.sample("pers_std", dist.HalfNormal(0.3))
    beta_std = numpyro.sample("beta_std", dist.HalfNormal(3))
    
    # Individual-level parameters
    with numpyro.plate("participants", choice.shape[1]):
        alpha_pos = numpyro.sample("alpha_pos", dist.TruncatedNormal(alpha_pos_mean, alpha_pos_std, low=0.01, high=0.99))
        alpha_neg = numpyro.sample("alpha_neg", dist.TruncatedNormal(alpha_neg_mean, alpha_neg_std, low=0.01, high=0.99))
        pers = numpyro.sample("pers", dist.TruncatedNormal(pers_mean, pers_std, low=0.01, high=0.99))
        beta = numpyro.sample("beta", dist.TruncatedNormal(beta_mean, beta_std, low=0.01, high=9.99))
        
    # # Basic bayesian inference (not hierarchical)
    # alpha_pos = numpyro.sample("alpha_pos", dist.Uniform(0., 1.))
    # alpha_neg = numpyro.sample("alpha_neg", dist.Uniform(0., 1.))
    # pers = numpyro.sample("pers", dist.Uniform(0., 1.))
    # beta = numpyro.sample("beta", dist.Uniform(0., 10.))

    def get_action_prob(q_values, pers_value):
        return jax.nn.sigmoid(beta * (q_values[:, 1] - q_values[:, 0] + pers_value))
    
    def get_pers_value(choice):
        return pers * (choice[:, 1] == 1)
    
    def update(carry, x):
        q_values = carry[:, :2]
        ch, rw = x[:, :2], x[:, 2]
        rw = rw.reshape(-1, 1)
        
        # Compute prediction errors for each outcome
        pe = (rw - q_values) * ch
        
        # Update Q-values
        lr = jnp.where(rw > 0.5, alpha_pos.reshape(-1, 1), alpha_neg.reshape(-1, 1))
        q_values = q_values + lr * pe
        
        # compute action probabilities
        action_prob = get_action_prob(q_values, get_pers_value(ch)).reshape(-1, 1)
        
        y = jnp.concatenate((q_values, action_prob), axis=-1)
        return y, y
    
    # Define initial Q-values and initialize the previous choice variable
    q_values = jnp.array((0.5, 0.5)).reshape(1, -1).repeat(choice.shape[1], axis=0)
    carry = jnp.concatenate((q_values, get_action_prob(q_values, get_pers_value(choice[0, :])).reshape(-1, 1)), axis=-1)
    xs = jnp.concatenate((choice[:-1], reward[:-1], choice[1:]), axis=-1)
    _, ys = jax.lax.scan(update, carry, xs)
    
    # Use numpyro.plate for sampling
    next_choices = choice[1:, :, -1]
    action_probs = ys[:, :, -1]
    
    with numpyro.plate("participants", choice.shape[1], dim=-1):
        with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
            numpyro.sample("obs", dist.Bernoulli(probs=action_probs), obs=next_choices)
            

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Performs a hierarchical bayesian parameter inference with numpyro.')
  
    parser.add_argument('--data', type=str, help='Dataset of a 2-armed bandit task with columns (session, choice, reward)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=500, help='Number of warmup samples (additional)')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    
    args = parser.parse_args()  
    
    # Prepare the data
    data = pd.read_csv(args.data)
    # get all different sessions
    sessions = data['session'].unique()
    # get maximum number of trials per session
    max_trials = data.groupby('session').size().max()
    # sort values into session-grouped arrays
    choices = np.zeros((len(sessions), max_trials, 2)) - 1
    rewards = np.zeros((len(sessions), max_trials, 1)) - 1
    for i, s in enumerate(sessions):
        choice = data[data['session'] == s]['choice'].values.astype(int)
        reward = data[data['session'] == s]['reward'].values
        choices[i, :len(choice)] = np.eye(2)[choice]
        rewards[i, :len(choice), 0] = reward
    
    # Run the model
    kernel = infer.NUTS(rl_model)
    mcmc = infer.MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains)
    mcmc.run(jax.random.PRNGKey(0), choice=jnp.array(choices.swapaxes(1, 0)), reward=jnp.array(rewards.swapaxes(1, 0)))
    # samples = mcmc.get_samples()
    # print(samples)

    # Convert to ArviZ InferenceData object
    inference_data = az.from_numpyro(mcmc)

    # save inference data
    az.to_netcdf(inference_data, 'traces.nc')