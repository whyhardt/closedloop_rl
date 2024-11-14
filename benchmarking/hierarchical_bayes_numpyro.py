import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.infer as infer
import jax.numpy as jnp
import jax
import pandas as pd
import arviz as az


def rl_model(choice: jnp.array, reward: jnp.array):
    n_sessions = choice.shape[0]
    
    # # Priors for group-level parameters
    # alpha_pos_mean = pyro.sample("alpha_pos_mean", dist.Normal(0, 1))
    # alpha_neg_mean = pyro.sample("alpha_neg_mean", dist.Normal(0, 1))
    # pers_mean = pyro.sample("pers_mean", dist.Normal(0, 1))
    # beta_mean = pyro.sample("beta_mean", dist.HalfNormal(1))
    
    # # Priors for individual-level variation (hierarchical)
    # alpha_pos_std = pyro.sample("alpha_pos_std", dist.HalfNormal(1))
    # alpha_neg_std = pyro.sample("alpha_neg_std", dist.HalfNormal(1))
    # pers_std = pyro.sample("pers_std", dist.HalfNormal(1))
    # beta_std = pyro.sample("beta_std", dist.HalfNormal(1))
    
    # # Individual-level parameters
    # with pyro.plate("participants", 1):  # n_sessions
    #     alpha_pos = pyro.sample("alpha_pos", dist.Normal(alpha_pos_mean, alpha_pos_std))
    #     alpha_neg = pyro.sample("alpha_neg", dist.Normal(alpha_neg_mean, alpha_neg_std))
    #     pers = pyro.sample("pers", dist.Normal(pers_mean, pers_std))
    #     beta = pyro.sample("beta", dist.Normal(beta_mean, beta_std))
    
    # Basic bayesian inference (not hierarchical)
    alpha_pos = numpyro.sample("alpha_pos", dist.Uniform(0.2, 0.3))
    alpha_neg = numpyro.sample("alpha_neg", dist.Uniform(0.2, 0.3))
    pers = numpyro.sample("pers", dist.Uniform(0.2, 0.3))
    beta = numpyro.sample("beta", dist.Uniform(2.8, 3.2))

    # Define initial Q-values and initialize the previous choice variable
    q_values = jnp.zeros((n_sessions, 2)) + 0.5
    prev_choice = jnp.zeros(n_sessions)  # stores the previous choice (0 or 1) for each participant

    for t in range(choice.shape[1]-1):
        ch = choice[:, t]
        ch_int = choice[:, t].astype(int)
        rw = reward[:, t]
        next_ch = choice[:, t+1]
        # mask = next_ch >= 0
        
        # Compute prediction errors for each outcome
        pe = rw - q_values[jnp.arange(n_sessions), ch_int]
        lr = jnp.where(rw == 1, alpha_pos, alpha_neg)

        # Update Q-values
        q_values.at[jnp.arange(n_sessions), ch_int].add(lr * pe)

        # Calculate action probabilities with perseverance effect
        perseverance_effect = ch == prev_choice #if t > 0 else jnp.zeros(n_sessions) # 1 if repeated choice, 0 otherwise
        logits = beta * (q_values[:, 1] - q_values[:, 0] + pers * perseverance_effect)
        action_prob = jax.nn.sigmoid(logits)

        # Likelihood of observed choices
        # with pyro.handlers.mask(mask=mask):
        #     pyro.sample(f"obs_{t}", dist.Bernoulli(probs=action_prob), obs=mask * next_ch)
        numpyro.sample(f"obs_{t}", dist.Bernoulli(probs=action_prob), obs=next_ch)
        
        # Update prev_choice to current choice for the next time step
        prev_choice = ch


# Prepare the data
data = pd.read_csv("data/data_rnn_a025_b30_f02_p025_ap025.csv")
# get all different sessions
sessions = data['session'].unique()
# get maximum number of trials per session
max_trials = data.groupby('session').size().max()
# sort values into session-grouped arrays
choices = np.zeros((len(sessions), max_trials)) - 1
rewards = np.zeros((len(sessions), max_trials)) - 1
for i, s in enumerate(sessions):
    choice = data[data['session'] == s]['choice'].values.astype(int)
    reward = data[data['session'] == s]['reward'].values
    choices[i, :len(choice)] = (choice)
    rewards[i, :len(choice)] = (reward)

# Run the model
kernel = infer.NUTS(rl_model)
mcmc = infer.MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), choice=jnp.array(choices), reward=jnp.array(rewards))
samples = mcmc.get_samples()
# print(samples)

# Convert to ArviZ InferenceData object
inference_data = az.from_numpyro(mcmc)

# Plot posterior distributions
az.plot_posterior(inference_data)