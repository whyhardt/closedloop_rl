import numpy as np
import matplotlib.pyplot as plt

from resources import bandits

def main(
    agent_1: bandits.AgentQ,
    agent_2: bandits.AgentQ,
    environment: bandits.Environment,
    normalize_q: bool = False,
):
    
    _, experiment_list_test = bandits.create_dataset(
        agent=agent_1,
        environment=environment,
        n_trials_per_session=200,
        n_sessions=1,
        )
    
    # Analysis
    session_id = 0

    choices = experiment_list_test[session_id].choices
    rewards = experiment_list_test[session_id].rewards

    list_probs = []
    list_qs = []

    # get q-values from first agent
    qs_test, probs_test = bandits.get_update_dynamics(experiment_list_test[session_id], agent_1)
    list_probs.append(np.expand_dims(probs_test, 0))
    list_qs.append(np.expand_dims(qs_test, 0))

    # get q-values from second agent
    qs_rnn, probs_rnn = bandits.get_update_dynamics(experiment_list_test[session_id], agent_2)
    list_probs.append(np.expand_dims(probs_rnn, 0))
    list_qs.append(np.expand_dims(qs_rnn, 0))

    colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

    # concatenate all choice probs and q-values
    probs = np.concatenate(list_probs, axis=0)
    qs = np.concatenate(list_qs, axis=0)

    # normalize q-values
    def normalize(qs):
      return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

    if normalize_q:
        qs = normalize(qs)

    fig, axs = plt.subplots(4, 1, figsize=(20, 10))

    reward_probs = np.stack([experiment_list_test[session_id].timeseries[:, i] for i in range(agent_1._n_actions)], axis=0)
    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=reward_probs,
        timeseries_name='Reward Probs',
        labels=[f'Arm {a}' for a in range(agent_1._n_actions)],
        color=['tab:purple', 'tab:cyan'],
        binary=True,
        fig_ax=(fig, axs[0]),
        )

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=probs[:, :, 0],
        timeseries_name='Choice Probs',
        color=colors,
        labels=['Agent 1', 'Agent 2'],
        binary=True,
        fig_ax=(fig, axs[1]),
        )

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=qs[:, :, 0],
        timeseries_name='Q Arm 0',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[2]),
        )

    # bandits.plot_session(
    #     compare=True,
    #     choices=choices,
    #     rewards=rewards,
    #     timeseries=qs[:, :, 1],
    #     timeseries_name='Q Arm 1',
    #     color=colors,
    #     binary=not non_binary_reward,
    #     fig_ax=(fig, axs[3]),
    #     )

    dqs_t = np.diff(qs, axis=1)

    bandits.plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=dqs_t[:, :, 0],
        timeseries_name='dQ/dt',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[3]),
        )

    # dqs_arms = normalize(-1*np.diff(qs, axis=2))

    # bandits.plot_session(
    #     compare=True,
    #     choices=choices,
    #     rewards=rewards,
    #     timeseries=dqs_arms[:, :, 0],
    #     timeseries_name='dQ/dActions',
    #     color=colors,
    #     binary=not non_binary_reward,
    #     fig_ax=(fig, axs[3]),
    #     )

    plt.show()
    
    
if __name__=='__main__':
    
    agent_1 = bandits.AgentQ(
        alpha=0.35,
        beta=3,
        forget_rate=0,
        perseveration_bias=0.,
        regret=False,
        confirmation_bias=0.,
    )
    
    agent_2 = bandits.AgentQ(
        alpha=0.35,
        beta=3,
        forget_rate=0,
        perseveration_bias=0.,
        regret=False,
        confirmation_bias=0.5,
    )
    
    env = bandits.EnvironmentBanditsDrift(sigma=0.2)
    
    main(agent_1, agent_2, env)
    