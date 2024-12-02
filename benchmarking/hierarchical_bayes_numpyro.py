import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import arviz as az
import argparse
import pickle
    

def rl_model(model: str, choice: jnp.array, reward: jnp.array, hierarchical: bool):
    
    if hierarchical:
        # Priors for group-level parameters
        alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Uniform(low=0.01, high=0.99)) if 'Ap' in model else 1
        alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Uniform(low=0.01, high=0.99)) if 'An' in model else -1
        alpha_c_mean = numpyro.sample("alpha_c_mean", dist.Uniform(low=0.01, high=0.99)) if 'Ac' in model else 1
        beta_c_mean = numpyro.sample("beta_c_mean", dist.Uniform(low=0.01, high=0.99)) if 'Bc' in model else 0
        beta_r_mean = numpyro.sample("beta_r_mean", dist.Uniform(low=0.01, high=9.99)) if 'Br' in model else 1
        
        # Priors for individual-level variation (hierarchical)
        alpha_pos_std = numpyro.sample("alpha_pos_std", dist.HalfNormal(0.3)) if 'Ap' in model else 0
        alpha_neg_std = numpyro.sample("alpha_neg_std", dist.HalfNormal(0.3)) if 'An' in model else 0
        alpha_c_std = numpyro.sample("alpha_c_std", dist.HalfNormal(0.3))  if 'Ac' in model else 0
        beta_c_std = numpyro.sample("beta_c_std", dist.HalfNormal(0.3)) if 'Bc' in model else 0
        beta_r_std = numpyro.sample("beta_r_std", dist.HalfNormal(3)) if 'Br' in model else 0
        
        # Individual-level parameters
        alpha_neg = None
        with numpyro.plate("participants", choice.shape[1]):
            alpha_pos = numpyro.sample("alpha_pos", dist.TruncatedNormal(alpha_pos_mean, alpha_pos_std, low=0.01, high=0.99))[:, None] if 'Ap' in model else 1
            if 'An' in model:
                alpha_neg = numpyro.sample("alpha_neg", dist.TruncatedNormal(alpha_neg_mean, alpha_neg_std, low=0.01, high=0.99))[:, None]
            alpha_c = numpyro.sample("alpha_c", dist.TruncatedNormal(alpha_c_mean, alpha_c_std, low=0.01, high=0.99))[:, None] if 'Ac' in model else 1
            beta_c = numpyro.sample("beta_c", dist.TruncatedNormal(beta_c_mean, beta_c_std, low=0.01, high=0.99)) if 'Bc' in model else 0
            beta_r = numpyro.sample("beta_r", dist.TruncatedNormal(beta_r_mean, beta_r_std, low=0.01, high=9.99)) if 'Br' in model else 1
        if 'An' not in model:
            alpha_neg = alpha_pos
    else:
        # Basic bayesian inference (not hierarchical)
        alpha_pos = numpyro.sample("alpha_pos", dist.Uniform(0., 1.)) if 'Ap' in model else 1
        alpha_neg = numpyro.sample("alpha_neg", dist.Uniform(0., 1.)) if 'An' in model else alpha_pos
        alpha_c = numpyro.sample("alpha_c", dist.Uniform(0., 1.)) if 'Ac' in model else 1
        beta_c = numpyro.sample("beta_c", dist.Uniform(0., 10.)) if 'Bc' in model else 0
        beta_r = numpyro.sample("beta_r", dist.Uniform(0., 10.)) if 'Br' in model else 1

    def get_action_prob(r_values, c_values):
        return jax.nn.sigmoid(beta_r * (r_values[:, 1] - r_values[:, 0]) + beta_c * (c_values[:, 1] - c_values[:, 0]))
    
    def update(carry, x):
        r_values = carry[:, :2]
        c_values = carry[:, 2:4]
        ch, rw = x[:, :2], x[:, 2][:, None]
        
        # Compute prediction errors for each outcome
        rpe = (rw - r_values) * ch
        cpe = ch - c_values
        
        # Update Q-values
        lr = jnp.where(rw > 0.5, alpha_pos, alpha_neg)
        r_values = r_values + lr * rpe
        # r_values = r_values + alpha_pos * rpe
        c_values = c_values + alpha_c * cpe
        
        # compute action probabilities
        action_prob = get_action_prob(r_values, c_values)[:, None]
        
        y = jnp.concatenate((r_values, c_values, action_prob), axis=-1)
        return y, y
    
    # Define initial Q-values and initialize the previous choice variable
    r_values = jnp.full((choice.shape[1], 2), 0.5)
    c_values = jnp.zeros((choice.shape[1], 2))
    carry = jnp.concatenate((r_values, c_values, get_action_prob(r_values, c_values)[:, None]), axis=-1)
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    # _, ys = jax.lax.scan(update, carry, xs)
    
    ys = jnp.zeros((choice.shape[0]-1, *carry.shape))
    for i in range(len(choice)-1):
        carry = update(carry, xs[i])[0]
        ys.at[i].add(carry)
        
    # Use numpyro.plate for sampling
    next_choices = choice[1:, :, -1]
    action_probs = ys[:, :, -1]
    
    if hierarchical:
        with numpyro.plate("participants", choice.shape[1], dim=-1):
            with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
                numpyro.sample("obs", dist.Bernoulli(probs=action_probs), obs=next_choices)
    else:
        numpyro.sample("obs", dist.Bernoulli(probs=action_probs.flatten()), obs=next_choices.flatten())



def main(file: str, model: str, num_samples: int, num_warmup: int, num_chains: int, hierarchical: bool, output_file: str):
    # Check model str
    valid_config = ['Ap', 'An', 'Ac', 'Bc', 'Br']
    model_checked = '' + model
    for c in valid_config:
        model_checked = model_checked.replace(c, '')
    if len(model_checked) > 0:
        raise ValueError(f'The provided model {model} is not supported. At least some part of the configuration ({model_checked}) is not valid. Valid configurations may include {valid_config}.')
    
    # Prepare the data
    data = pd.read_csv(file)
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
    mcmc = infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.PRNGKey(0), model=model, choice=jnp.array(choices.swapaxes(1, 0)), reward=jnp.array(rewards.swapaxes(1, 0)), hierarchical=hierarchical)

    with open(output_file.split('.')[0] + '_' + model + '.nc', 'wb') as file:
        pickle.dump(mcmc, file)
    
    return mcmc

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Performs a hierarchical bayesian parameter inference with numpyro.')
  
    parser.add_argument('--file', type=str, help='Dataset of a 2-armed bandit task with columns (session, choice, reward)')
    parser.add_argument('--model', type=str, default='ApAnAcBcBr', help='Model configuration (Ap: learning rate for positive outcomes, An: learning rate for negative outcomes, Ac: learning rate for choice-based value, Bc: Importance of choice-based values, Br: Importance and inverse noise termperature for reward-based values)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of MCMC samples')
    parser.add_argument('--num_warmup', type=int, default=500, help='Number of warmup samples (additional)')
    parser.add_argument('--num_chains', type=int, default=1, help='Number of chains')
    parser.add_argument('--hierarchical', action='store_true', help='Whether to do hierarchical inference')
    parser.add_argument('--output_file', type=str, default='benchmarking/params/traces.nc', help='Number of chains')
    
    args = parser.parse_args()

    # with jax.default_device(jax.devices("cpu")[0]):
    main(args.file, args.model, args.num_samples, args.num_warmup, args.num_chains, args.hierarchical, args.output_file)
    