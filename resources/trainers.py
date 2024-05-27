import pickle
from typing import Iterable

import numpy as np
import optax
import pysindy as ps
import sklearn

from resources import rnn_utils, bandits
from resources.models import AgentSindy, AgentNetwork_VisibleState


class RNNSindyEstimator(sklearn.base.BaseEstimator):
    def __init__(self, params_path='rnn_params.pkl', rnn_fun=None, train=True, load=False,
                 loss_function='categorical',
                 n_trials_per_session=200, n_sessions=220,
                 convergence_thresh=1e-5, n_steps_max=10000,
                 habit_weight=0.0, n_actions=2,
                 agent=None, get_choices=False, threshold=0.01,
                 library=None, ensemble=False, library_ensemble=False, non_binary_reward=False,
                 alpha=0.1, beta=3, dt=1.0, sigma=0.1,
                 normalize=False, verbose=False):
        self.params_path = params_path
        self.rnn_fun = rnn_fun
        self.train = train
        self.load = load
        self.loss_function = loss_function
        self.n_trials_per_session = n_trials_per_session
        self.n_sessions = n_sessions
        self.convergence_thresh = convergence_thresh
        self.n_steps_max = n_steps_max
        self.habit_weight = habit_weight
        self.n_actions = n_actions
        self.agent = agent
        self.get_choices = get_choices
        self.threshold = threshold
        self.library = library
        self.ensemble = ensemble
        self.library_ensemble = library_ensemble
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.non_binary_reward = non_binary_reward
        self.dt = dt
        self.normalize = normalize
        self.verbose = verbose

        self.rnn_trainer = None
        self.sindy_trainer = None
        self.experiment_list_hybrnn = None
        self.dataset_hybrnn = None
        self.environment = None

    def fit(self, X):
        # Train RNN
        dataset_train, experiment_list_train = X
        self.rnn_trainer = RNNTrainer(params_path='rnn_params.pkl', model_fun=self.rnn_fun, train=self.train,
                                      load=self.load,
                                      loss_function=self.loss_function, convergence_thresh=self.convergence_thresh,
                                      n_steps_max=self.n_steps_max, habit_weight=self.habit_weight,
                                      n_actions=self.n_actions)
        self.rnn_trainer.fit(dataset_train)

        # Synthesize dataset with RNN
        self.environment = bandits.EnvironmentBanditsDrift(sigma=self.sigma, n_actions=self.n_actions,
                                                           non_binary_rewards=self.non_binary_reward)

        hybrnn_agent = self.rnn_trainer.visible_agent
        self.dataset_hybrnn, self.experiment_list_hybrnn = bandits.create_dataset(hybrnn_agent, self.environment,
                                                                                  self.n_trials_per_session,
                                                                                  self.n_sessions)

        # Train SINDy
        self.sindy_trainer = SindyTrainer(agent=self.agent, n_actions=self.n_actions, get_choices=self.get_choices,
                                          threshold=self.threshold, library=self.library, ensemble=self.ensemble,
                                          library_ensemble=self.library_ensemble, alpha=self.alpha, beta=self.beta,
                                          dt=self.dt, normalize=self.normalize, verbose=self.verbose)
        self.sindy_trainer.fit(self.experiment_list_hybrnn)

        return self


class RNNTrainer(sklearn.base.BaseEstimator):
    def __init__(self, params_path='rnn_params.pkl', model_fun=None, train=True, load=False,
                 loss_function='categorical',
                 convergence_thresh=1e-5, n_steps_max=10000, habit_weight=0.0, n_actions=2):
        self.params_path = params_path
        self.train = train
        self.load = load
        self.loss_function = loss_function
        self.optimizer = optax.adam(learning_rate=1e-3)
        self.model_fun = model_fun
        self.convergence_thresh = convergence_thresh
        self.n_steps_max = n_steps_max
        self.habit_weight = habit_weight
        self.n_actions = n_actions
        self.rnn_params = None
        self.opt_state = None
        self.visible_agent = None

    def load_parameters(self):
        try:
            with open(self.params_path, 'rb') as f:
                saved_params = pickle.load(f)
            self.rnn_params, self.opt_state = saved_params[0], saved_params[1]
            print('Loaded parameters.')
        except FileNotFoundError:
            print('No parameters found to load.')

    def save_parameters(self):
        with open(self.params_path, 'wb') as f:
            pickle.dump((self.rnn_params, self.opt_state), f)
        print('Parameters saved.')

    def fit(self, X):
        if self.train:
            if self.load:
                self.load_parameters()
            else:
                if self.model_fun is None:
                    raise ValueError('Model function must be provided for training.')
                self.rnn_params, self.opt_state = None, None

            print('Training the hybrid RNN...')
            self.rnn_params, self.opt_state, _ = rnn_utils.fit_model(
                model_fun=self.model_fun,
                dataset=X,
                optimizer=self.optimizer,
                optimizer_state=self.opt_state,
                model_params=self.rnn_params,
                loss_fun=self.loss_function,
                convergence_thresh=self.convergence_thresh,
                n_steps_max=self.n_steps_max
            )

            self.save_parameters()

            self.visible_agent = AgentNetwork_VisibleState(self.model_fun, self.rnn_params,
                                                           habit=self.habit_weight == 1, n_actions=self.n_actions)
        return self

    def execute(self, dataset_train):
        if self.train:
            self.fit(dataset_train)
        else:
            self.load_parameters()

    def get_rnn_params(self):
        return self.rnn_params


class SindyTrainer(sklearn.base.BaseEstimator):
    def __init__(self, agent=None, n_actions=2, get_choices=False, threshold=0.01, library=None, ensemble=False,
                 library_ensemble=False, alpha=0.1, beta=3, dt=1.0, normalize=False, verbose=False):
        self.agent = agent
        self.n_actions = n_actions
        self.get_choices = get_choices
        self.threshold = threshold
        self.library = library
        self.ensemble = ensemble
        self.library_ensemble = library_ensemble
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.normalize = normalize
        self.verbose = verbose
        self.sindy = None
        self.sindyagent = None

    def fit(self, X):
        """
        Fit SINDy model to dataset
        Args:
            X: array-like, shape (n_sessions, n_trials_per_session, n_actions) corresponding to experiment list

        Returns: self

        """
        if self.agent is None:
            raise ValueError('Agent must be provided for training.')
        if self.library is None:
            raise ValueError('Library must be provided for training.')

        x_train, control, feature_names = self.make_sindy_data(X, self.agent,
                                                               get_choices=self.get_choices)
        if self.normalize:
            # scale q-values between 0 and 1 for more realistic dynamics
            x_max = np.max(np.stack(x_train, axis=0))
            x_min = np.min(np.stack(x_train, axis=0))
            print(f'Dataset characteristics: max={x_max}, min={x_min}')
            x_train = [(x - x_min) / (x_max - x_min) for x in x_train]

        self.sindy = ps.SINDy(
            optimizer=ps.STLSQ(threshold=self.threshold, verbose=self.verbose, alpha=0.1),
            feature_library=self.library,
            discrete_time=True,
            feature_names=feature_names,
        )
        self.sindy.fit(x_train, t=self.dt, u=control, ensemble=self.ensemble, library_ensemble=self.library_ensemble,
                       multiple_trajectories=True)
        self.sindy.print()
        sparsity_index = np.sum(self.sindy.coefficients() < self.threshold) / self.sindy.coefficients().size
        print(f'Sparsity index: {sparsity_index}')

        # set new sindy update rule and synthesize new dataset
        if not self.get_choices:
            update_rule_datasindy = lambda q, choice, reward: \
                self.sindy.simulate(q[choice], t=2, u=np.array(reward).reshape(1, 1))[-1]
        else:
            update_rule_datasindy = lambda q, choice, reward: \
                self.sindy.simulate(q, t=2, u=np.array([choice, reward]).reshape(1, 2))[-1]

        self.sindyagent = AgentSindy(alpha=0, beta=1, n_actions=self.n_actions)
        self.sindyagent.set_update_rule(update_rule_datasindy)

        return self

    def make_sindy_data(self,
                        dataset,
                        agent: bandits.AgentQ,
                        sessions=-1,
                        get_choices=True
                        ):
        # Get training data for SINDy
        # put all relevant signals in x_train

        if not isinstance(sessions, Iterable) and sessions == -1:
            # use all sessions
            sessions = np.arange(len(dataset))
        else:
            # use only the specified sessions
            sessions = np.array(sessions)

        if get_choices:
            n_control = 2
        else:
            n_control = 1

        choices = np.stack([dataset[i].choices for i in sessions], axis=0)
        rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
        qs = np.stack([dataset[i].q for i in sessions], axis=0)

        if not get_choices:
            raise NotImplementedError('Only get_choices=True is implemented right now.')
            n_sessions = qs.shape[0]
            n_trials = qs.shape[1] * qs.shape[2]
            qs_all = np.zeros((n_sessions, n_trials))
            r_all = np.zeros((n_sessions, n_trials))
            c_all = None
            # concatenate the data of all arms into one array for more training data
            index_end_last_arm = 0
            for index_arm in range(agent._n_actions):
                index = np.where(choices == index_arm)[0]
                r_all[index_end_last_arm:index_end_last_arm + len(index)] = rewards[index]
                qs_all[index_end_last_arm:index_end_last_arm + len(index)] = qs[index, index_arm].reshape(-1, 1)
                index_end_last_arm += len(index)
        else:
            choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
            for sess in sessions:
                # one-hot encode choices
                choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
                # concatenate all qs values of one sessions along the trial dimension
                qs_all = np.concatenate(
                    [np.stack([np.expand_dims(qs_sess[:, i], axis=-1) for i in range(agent._n_actions)], axis=0) for
                     qs_sess
                     in qs], axis=0)
                c_all = np.concatenate(
                    [np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0) for c_sess in choices_oh],
                    axis=0)
                r_all = np.concatenate(
                    [np.stack([r_sess for _ in range(agent._n_actions)], axis=0) for r_sess in rewards],
                    axis=0)

        # get observed dynamics
        x_train = qs_all
        feature_names = ['q']

        # get control
        control_names = []
        control = np.zeros((*x_train.shape[:-1], n_control))
        if get_choices:
            control[:, :, 0] = c_all
            control_names += ['c']
        control[:, :, n_control - 1] = r_all
        control_names += ['r']

        feature_names += control_names

        print(f'Shape of Q-Values is: {x_train.shape}')
        print(f'Shape of control parameters is: {control.shape}')
        print(f'Feature names are: {feature_names}')

        # make x_train and control sequences instead of arrays
        x_train = [x_train_sess for x_train_sess in x_train]
        control = [control_sess for control_sess in control]

        return x_train, control, feature_names
