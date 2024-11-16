import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import arviz as az
    

def rl_model(choice: jnp.array, reward: jnp.array):
    
    # # Priors for group-level parameters
    # alpha_pos_mean = numpyro.sample("alpha_pos_mean", dist.Normal(0, 1))
    # alpha_neg_mean = numpyro.sample("alpha_neg_mean", dist.Normal(0, 1))
    # pers_mean = numpyro.sample("pers_mean", dist.Normal(0, 1))
    # beta_mean = numpyro.sample("beta_mean", dist.HalfNormal(1))
    
    # # Priors for individual-level variation (hierarchical)
    # alpha_pos_std = numpyro.sample("alpha_pos_std", dist.HalfNormal(1))
    # alpha_neg_std = numpyro.sample("alpha_neg_std", dist.HalfNormal(1))
    # pers_std = numpyro.sample("pers_std", dist.HalfNormal(1))
    # beta_std = numpyro.sample("beta_std", dist.HalfNormal(1))
    
    # # Individual-level parameters
    # with numpyro.plate("participants", choice.shape[1]):
    #     alpha_pos = numpyro.sample("alpha_pos", dist.Normal(alpha_pos_mean, alpha_pos_std))
    #     alpha_neg = numpyro.sample("alpha_neg", dist.Normal(alpha_neg_mean, alpha_neg_std))
    #     pers = numpyro.sample("pers", dist.Normal(pers_mean, pers_std))
    #     beta = numpyro.sample("beta", dist.Normal(beta_mean, beta_std))
    
    # Basic bayesian inference (not hierarchical)
    alpha_pos = numpyro.sample("alpha_pos", dist.Uniform(0., 1.))
    alpha_neg = numpyro.sample("alpha_neg", dist.Uniform(0., 1.))
    pers = numpyro.sample("pers", dist.Uniform(0., 1.))
    beta = numpyro.sample("beta", dist.Uniform(0., 10.))

    def get_action_prob(q_values, pers_value):
        return jax.nn.sigmoid(beta * (q_values[:, 1] - q_values[:, 0] + pers_value))
    
    def get_pers_value(choice):
        return pers * (choice[:, 1] == 1)
    
    def update(carry, x):
        q_values = carry[:, :2]
        ch, rw = x[:, :2], x[:, 2]
        rw = rw.reshape(-1, 1)
        
        # Compute prediction errors for each outcome
        lr = jnp.where(rw > 0.5, alpha_pos, alpha_neg)
        pe = (rw - q_values) * ch
        
        # Update Q-values
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
    with numpyro.plate("time_steps", xs.shape[0] * xs.shape[1]):
        numpyro.sample("obs", dist.Bernoulli(probs=action_probs.flatten()), obs=next_choices.flatten())


# Prepare the data
data = pd.read_csv("data/data_rnn_a025_b30_ap025.csv")
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
mcmc = infer.MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=1)
mcmc.run(jax.random.PRNGKey(0), choice=jnp.array(choices.swapaxes(1, 0)), reward=jnp.array(rewards.swapaxes(1, 0)))
samples = mcmc.get_samples()
# print(samples)

# Convert to ArviZ InferenceData object
inference_data = az.from_numpyro(mcmc)

# Plot posterior distributions
az.plot_trace(inference_data)
import matplotlib.pyplot as plt 
plt.show()