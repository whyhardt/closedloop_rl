import sys
import os

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import AgentQ, EnvironmentBanditsDrift, EnvironmentBanditsSwitch, plot_session, create_dataset, get_update_dynamics
from utils.plotting import session

agent1 = AgentQ(
    beta_reward=3.,
    alpha_reward=0.25,
    alpha_penalty=-1,
    alpha_counterfactual=0.5,
    beta_choice=0.,
    alpha_choice=0.,
    forget_rate=0.,
    confirmation_bias=0.,
    )

agent2 = AgentQ(
    beta_reward=3.,
    alpha_reward=0.25,
    alpha_penalty=-1,
    alpha_counterfactual=0.5,
    beta_choice=3.,
    alpha_choice=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    )

# env = EnvironmentBanditsSwitch(0.05, reward_prob_high=1.0, reward_prob_low=0.5)
env = EnvironmentBanditsDrift(0.1)
trajectory = create_dataset(agent1, env, 256, 1)[1][0]
agents = {'groundtruth': agent1, 'rnn': agent2}
fig, axs = session(agents, trajectory)
plt.show()