from resources import bandits


class SyntheticDataset:
    def __init__(self, agent, n_sessions, n_trials_per_session, sigma=0.1, non_binary_reward=False):
        self.agent = agent

        self.n_sessions = n_sessions
        self.n_trials_per_session = n_trials_per_session
        self.sigma = sigma
        self.non_binary_reward = non_binary_reward

        self.environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=agent._n_actions,
                                                           non_binary_rewards=non_binary_reward)

        self.dataset_train = None
        self.dataset_test = None
        self.experiment_list_train = None
        self.experiment_list_test = None

        self.generate()

    def generate(self):
        self.dataset_train, self.experiment_list_train = self.generate_data()
        self.dataset_test, self.experiment_list_test = self.generate_data()

    def generate_data(self):
        return bandits.create_dataset(
            agent=self.agent,
            environment=self.environment,
            n_trials_per_session=self.n_trials_per_session,
            n_sessions=self.n_sessions
        )
