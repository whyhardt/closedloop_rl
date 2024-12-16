import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import argparse
import pickle
    
# @jax.jit(static_argnames=['model','hierarchical'])
def rl_model(model, choice, reward, hierarchical):

    def scaled_beta(a, b, low, high):
        return dist.TransformedDistribution(
            dist.Beta(a, b),  # Beta distribution in [0, 1]
            dist.transforms.AffineTransform(low, high - low)  # Scale to [0, 10]
            )
    
    if hierarchical == 1:
        # Priors for group-level parameters
        # alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Uniform(low=0.01, high=0.99)) if model[0]==1 else 1
        # alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Uniform(low=0.01, high=0.99)) if model[1]==1 else -1
        # alpha_c_mean = numpyro.sample("alpha_c_mean", dist.Uniform(low=0.01, high=0.99)) if model[2]==1 else 1
        # beta_c_mean = numpyro.sample("beta_c_mean", dist.Uniform(low=0.01, high=0.99)) if model[3]==1 else 0
        # beta_r_mean = numpyro.sample("beta_r_mean", dist.Uniform(low=0.01, high=9.99)) if model[4]==1 in model else 1
        alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Beta(2, 2)) if model[0]==1 else 1
        alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Beta(2, 2)) if model[1]==1 else -1
        alpha_c_mean = numpyro.sample("alpha_c_mean", dist.Beta(2, 2)) if model[2]==1 else 1
        beta_c_mean = numpyro.sample("beta_c_mean", scaled_beta(2, 2, 0, 10)) if model[3]==1 else 0
        beta_r_mean = numpyro.sample("beta_r_mean", scaled_beta(2, 2, 0, 10)) if model[4]==1 in model else 1
        
        # Priors for individual-level variation (hierarchical)
        # alpha_pos_std = numpyro.sample("alpha_pos_std", dist.HalfNormal(0.3)) if model[0]==1 else 0
        # alpha_neg_std = numpyro.sample("alpha_neg_std", dist.HalfNormal(0.3)) if model[1]==1 else 0
        # alpha_c_std = numpyro.sample("alpha_c_std", dist.HalfNormal(0.3))  if model[2]==1 else 0
        # beta_c_std = numpyro.sample("beta_c_std", dist.HalfNormal(0.3)) if model[3]==1 else 0
        # beta_r_std = numpyro.sample("beta_r_std", dist.HalfNormal(3)) if model[4]==1 else 0
        alpha_pos_std = numpyro.sample("alpha_pos_std", dist.Beta(2, 2)) if model[0]==1 else 0
        alpha_neg_std = numpyro.sample("alpha_neg_std", dist.Beta(2, 2)) if model[1]==1 else 0
        alpha_c_std = numpyro.sample("alpha_c_std", dist.Beta(2, 2))  if model[2]==1 else 0
        beta_c_std = numpyro.sample("beta_c_std", scaled_beta(2, 2, 0, 10)) if model[3]==1 else 0
        beta_r_std = numpyro.sample("beta_r_std", scaled_beta(2, 2, 0, 10)) if model[4]==1 else 0

        # Individual-level parameters
        alpha_neg = None
        with numpyro.plate("participants", choice.shape[1]):
            alpha_pos = numpyro.sample("alpha_pos", dist.TruncatedNormal(alpha_pos_mean, alpha_pos_std, low=0.01, high=0.99))[:, None] if model[0]==1 else 1
            if model[1]==1:
                alpha_neg = numpyro.sample("alpha_neg", dist.TruncatedNormal(alpha_neg_mean, alpha_neg_std, low=0.01, high=0.99))[:, None]
            alpha_c = numpyro.sample("alpha_c", dist.TruncatedNormal(alpha_c_mean, alpha_c_std, low=0.01, high=0.99))[:, None] if model[2]==1 else 1
            beta_c = numpyro.sample("beta_c", dist.TruncatedNormal(beta_c_mean, beta_c_std, low=0.01, high=0.99)) if model[3]==1 else 0
            beta_r = numpyro.sample("beta_r", dist.TruncatedNormal(beta_r_mean, beta_r_std, low=0.01, high=9.99)) if model[4]==1 else 1
            
        if model[1]==0:
            alpha_neg = alpha_pos
    else:
        # Basic bayesian inference (not hierarchical)
        alpha_pos = numpyro.sample("alpha_pos", dist.Beta(2, 2)) if model[0]==1 else 1
        alpha_neg = numpyro.sample("alpha_neg", dist.Beta(2, 2)) if model[1]==1 else alpha_pos
        alpha_c = numpyro.sample("alpha_c", dist.Beta(2, 2)) if model[2]==1 else 1
        beta_c = numpyro.sample("beta_c", scaled_beta(2, 2, 0, 10)) if model[3]==1 else 0
        beta_r = numpyro.sample("beta_r", scaled_beta(2, 2, 0, 10)) if model[4]==1 else 1
    
    def update(carry, x):#, alpha_pos, alpha_neg, alpha_c, beta_r, beta_c):
        r_values = carry[0]
        c_values = carry[1]
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
        r_diff = r_values[:, 1] - r_values[:, 0]
        c_diff = c_values[:, 1] - c_values[:, 0]
        action_prob = jax.nn.sigmoid(beta_r * r_diff + beta_c * c_diff)
        # action_prob = get_action_prob(r_values, c_values)[:, None]
        
        # y = jnp.concatenate((r_values, c_values, action_prob), axis=-1)
        
        return (r_values, c_values), action_prob
    
    # Define initial Q-values and initialize the previous choice variable
    r_values = jnp.full((choice.shape[1], 2), 0.5)
    c_values = jnp.zeros((choice.shape[1], 2))
    xs = jnp.concatenate((choice[:-1], reward[:-1]), axis=-1)
    # _, ys = jax.lax.scan(update, (r_values, c_values), xs)
    
    ys = jnp.zeros((choice.shape[0]-1, r_values.shape[0]))
    carry = (r_values, c_values)
    for i in range(len(choice)-1):
        carry, y = update(carry, xs[i])
        ys = ys.at[i].set(y)
    
    # Use numpyro.plate for sampling
    next_choices = choice[1:, :, -1]
    valid_mask = ys != -1
    # Apply the mask to the observations
    if hierarchical == 1:
        with numpyro.handlers.mask(mask=valid_mask):
            with numpyro.plate("participants", choice.shape[1], dim=-1):
                with numpyro.plate("time_steps", choice.shape[0] - 1, dim=-2):
                    numpyro.sample("obs", dist.Bernoulli(probs=ys), obs=next_choices)
    else:
        with numpyro.handlers.mask(mask=valid_mask.flatten()):
            numpyro.sample("obs", dist.Bernoulli(probs=ys.flatten()), obs=next_choices.flatten())


def encode_model_name(model: str, model_parts: list) -> np.ndarray:
    enc = np.zeros((len(model_parts),))
    for i in range(len(model_parts)):
        if model_parts[i] in model:
            enc[i] = 1
    return enc


def main(file: str, model: str, num_samples: int, num_warmup: int, num_chains: int, hierarchical: bool, output_file: str, checkpoint: bool):
    # set output file
    output_file = output_file.split('.')[0] + '_' + model + '.nc'
    
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
    numpyro.set_host_device_count(num_chains)
    print(f'Number of devices: {jax.device_count()}')
    kernel = infer.NUTS(rl_model)
    if checkpoint and num_warmup > 0:
        print(f'Checkpoint was set but num_warmup>0 ({num_warmup}). Setting num_warmup=0.')
        num_warmup = 0
    mcmc = infer.MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    print('Initialized MCMC model.')
    if checkpoint:
        with open(output_file, 'rb') as file:
            checkpoint = pickle.load(file)
        mcmc.post_warmup_state = checkpoint.last_state
        rng_key = mcmc.post_warmup_state.rng_key
        print('Checkpoint loaded.')
    else:
        rng_key = jax.random.PRNGKey(0)
    mcmc.run(rng_key, model=tuple(encode_model_name(model, valid_config)), choice=jnp.array(choices.swapaxes(1, 0)), reward=jnp.array(rewards.swapaxes(1, 0)), hierarchical=hierarchical)

    with open(output_file, 'wb') as file:
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
    parser.add_argument('--checkpoint', action='store_true', help='Whether to load the specified output file as a checkpoint')
    
    args = parser.parse_args()

    # with jax.default_device(jax.devices("cpu")[0]):
    main(args.file, args.model, args.num_samples, args.num_warmup, args.num_chains, args.hierarchical, args.output_file, args.checkpoint)
    