import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable

sys.path.append('resources')
from resources.rnn import RLRNN, EnsembleRNN
from resources.bandits import AgentQ, AgentSindy, AgentNetwork, EnvironmentBanditsDrift, EnvironmentBanditsSwitch, plot_session, get_update_dynamics, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, check_library_setup, bandit_loss
from resources.rnn_utils import parameter_file_naming
from resources.sindy_training import fit_model
from utils.convert_dataset import to_datasetrnn

warnings.filterwarnings("ignore")

def main(
    model = None,
    data = None,
    
    # generated training dataset parameters
    n_trials_per_session = 200,
    n_sessions = 1,
    
    # sindy parameters
    threshold = 0.03,
    polynomial_degree = 1,
    regularization = 1e-2,
    
    # rnn parameters
    hidden_size = 8,
    
    # ground truth parameters
    alpha = 0.25,
    regret = False,
    confirmation_bias = False,
    forget_rate = 0.,
    perseveration_bias = 0.,
    beta = 3,
    directed_exploration_bias = 0.,
    undirected_exploration_bias = 0.,
    reward_prediction_error: Callable = None,
    
    # environment parameters
    n_actions = 2,
    sigma = .2,
    non_binary_reward = False,
    correlated_reward = False,
    
    analysis=False,
    ):

    # rnn parameters
    use_lstm = False
    voting_type = EnsembleRNN.MEDIAN

    # tracked variables in the RNN
    # single_entries = {
    #     'xLR': ['xLR_0', 'xLR_1', 'xLR_2', 'xLR_3', 'xLR_4', 'xLR_5', 'xLR_6', 'xLR_7']
    # }
    z_train_list = ['xLR', 'xQf', 'xH', 'xHf']#, 'xU', 'xUf', 'xB']
    control_list = ['ca', 'cr', 'cp', 'ca_repeat', 'cQ']#, 'cU_0', 'cU_1']
    sindy_feature_list = z_train_list + control_list

    # library setup aka which terms are allowed as control inputs in each SINDy model
    # key is the SINDy submodel name, value is a list of allowed control inputs
    library_setup = {
        'xLR': ['cQ', 'cr', 'cp'],
        'xQf': [],
        'xH': ['ca_repeat'],
        'xHf': [],
        # 'xU': ['cQ', 'cr'],
        # 'xUf': [],
        # 'xB': ['cU_0', 'cU_1'],
    }

    # data-filter setup aka which samples are allowed as training samples in each SINDy model based on the given filter condition
    # key is the SINDy submodel name, value is a list with a triplet of values: 
    #   1. str: feature name to be used as a filter
    #   2. numeric: the numeric filter condition
    #   3. bool: remove feature from control inputs
    # Can also be a list of list of triplets for multiple filter conditions
    # Example:
    # 'xQf': ['ca', 0, True] means that only samples where the feature 'ca' is 0 are used for training the SINDy model 'xQf' and the control parameter 'ca' is removed for training the model
    datafilter_setup = {
        'xLR': ['ca', 1, True],
        'xQf': ['ca', 0, True],
        'xH': ['ca', 1, True],
        'xHf': ['ca', 0, True],
        # 'xU': ['ca', 1, True],
        # 'xUf': ['ca', 0, True],
    }

    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')

    # set up rnn agent and expose q-values to train sindy
    if model is None:
        params_path = parameter_file_naming('params/params', use_lstm, alpha, beta, forget_rate, perseveration_bias, regret, confirmation_bias, directed_exploration_bias, undirected_exploration_bias, verbose=True)
    else:
        params_path = model
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))['model']
    rnn = RLRNN(n_actions, hidden_size, 0.5, list_sindy_signals=sindy_feature_list)
    if isinstance(state_dict, dict):
        rnn.load_state_dict(state_dict)
    elif isinstance(state_dict, list):
        print('Loading ensemble model...')
        model_list = []
        for state_dict_i in state_dict:
            model_list.append(deepcopy(rnn))
            model_list[-1].load_state_dict(state_dict_i)
        rnn = EnsembleRNN(model_list, voting_type=voting_type)
    agent_rnn = AgentNetwork(rnn, n_actions, deterministic=True)

    # set up ground truth agent and environment
    # environment = EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_reward=non_binary_reward, correlated_reward=correlated_reward)
    # environment = EnvironmentBanditsSwitch(sigma)
    if data is None:
        agent = AgentQ(n_actions, alpha, beta, forget_rate, perseveration_bias, regret, confirmation_bias, directed_exploration_bias, undirected_exploration_bias)
        if reward_prediction_error is not None:
            agent.set_reward_prediction_error(reward_prediction_error)
            # _, experiment_list_test = create_dataset_bandits(agent_rnn, environment, 200, 1)
            # _, experiment_list_train = create_dataset_bandits(agent_rnn, environment, n_trials_per_session, n_sessions)
    else:
        _, experiment_list_train = to_datasetrnn(data)
        experiment_list_test = [experiment_list_train[-1]]
        experiment_list_train = experiment_list_train[:-1]            

    # create dataset for sindy training, fit sindy, set up sindy agent
    print(f'SINDy Beta: {agent_rnn._model.beta.item():.2f}')
    z_train, control, feature_names = create_dataset(agent_rnn, experiment_list_train, n_trials_per_session, n_sessions, clear_offset=False, shuffle=True, trimming=False)
    sindy_models = fit_model(z_train, control, feature_names, polynomial_degree, library_setup, datafilter_setup, True, False, threshold, regularization)
    agent_sindy = AgentSindy(sindy_models, n_actions, agent_rnn._model.beta.item(), True)

    print('Calculating RNN and SINDy loss in X...', end='\r')
    test_loss_rnn_x = bandit_loss(agent_rnn, experiment_list_test, coordinates="x")
    test_loss_sindy_x = bandit_loss(agent_sindy, experiment_list_test, coordinates="x")
    print(f'RNN Loss in X (predicting behavior; Target: Subject): {test_loss_rnn_x}')
    print(f'SINDy Loss in X (predicting behavior; Target: Subject): {test_loss_sindy_x}')
    # test_loss_sindy_z = bandit_loss(agent_sindy, experiment_list_train[:10], coordinates="z")
    # test_loss_rnn_z = bandit_loss(agent_rnn, experiment_list_test[:10], coordinates="z")
    # print(f'RNN Loss in Z (comparing choice probabilities; Target: Subject): {test_loss_rnn_z}')
    # print(f'SINDy Loss in Z (comparing choice probabilities; Target: RNN): {test_loss_sindy_z}')
    
    # --------------------------------------------------------------
    # Analysis
    # --------------------------------------------------------------

    if analysis:
        labels = ['Ground Truth', 'RNN', 'SINDy'] if data is None else ['RNN', 'SINDy']
        experiment_test = experiment_list_test[0]
        choices = experiment_test.choices
        rewards = experiment_test.rewards

        list_probs = []
        list_Qs = []
        list_qs = []
        list_hs = []
        list_us = []
        list_bs = []

        # get q-values from groundtruth
        if data is None:
            qs_test, probs_test, _ = get_update_dynamics(experiment_list_test[0], agent)
            list_probs.append(np.expand_dims(probs_test, 0))
            list_Qs.append(np.expand_dims(qs_test[0], 0))
            list_qs.append(np.expand_dims(qs_test[1], 0))
            list_hs.append(np.expand_dims(qs_test[2], 0))
            list_us.append(np.expand_dims(qs_test[3], 0))
            list_bs.append(np.expand_dims(qs_test[4], 0))

        # get q-values from trained rnn
        qs_rnn, probs_rnn, _ = get_update_dynamics(experiment_list_test[0], agent_rnn)
        list_probs.append(np.expand_dims(probs_rnn, 0))
        list_Qs.append(np.expand_dims(qs_rnn[0], 0))
        list_qs.append(np.expand_dims(qs_rnn[1], 0))
        list_hs.append(np.expand_dims(qs_rnn[2], 0))
        list_us.append(np.expand_dims(qs_rnn[3], 0))
        list_bs.append(np.expand_dims(qs_rnn[4], 0))
        
        # get q-values from trained sindy
        qs_sindy, probs_sindy, _ = get_update_dynamics(experiment_list_test[0], agent_sindy)
        list_probs.append(np.expand_dims(probs_sindy, 0))
        list_Qs.append(np.expand_dims(qs_sindy[0], 0))
        list_qs.append(np.expand_dims(qs_sindy[1], 0))
        list_hs.append(np.expand_dims(qs_sindy[2], 0))
        list_us.append(np.expand_dims(qs_sindy[3], 0))
        list_bs.append(np.expand_dims(qs_sindy[4], 0))

        colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

        # concatenate all choice probs and q-values
        probs = np.concatenate(list_probs, axis=0)
        Qs = np.concatenate(list_Qs, axis=0)
        qs = np.concatenate(list_qs, axis=0)
        hs = np.concatenate(list_hs, axis=0)
        us = np.concatenate(list_us, axis=0)
        bs = np.concatenate(list_bs, axis=0)

        colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

        # normalize q-values
        def normalize(qs):
            return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

        # qs = normalize(qs)

        fig, axs = plt.subplots(7, 1, figsize=(20, 10))

        reward_probs = np.stack([experiment_list_test[0].reward_probabilities[:, i] for i in range(n_actions)], axis=0)
        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=reward_probs,
            timeseries_name='p(R)',
            labels=[f'Arm {a}' for a in range(n_actions)],
            color=['tab:purple', 'tab:cyan'],
            binary=not non_binary_reward,
            fig_ax=(fig, axs[0]),
            )

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=probs[:, :, 0],
            timeseries_name='p(A)',
            color=colors,
            labels=labels,
            binary=not non_binary_reward,
            fig_ax=(fig, axs[1]),
            )

        plot_session(
        compare=True,
        choices=choices,
        rewards=rewards,
        timeseries=Qs[:, :, 0],
        timeseries_name='Q',
        color=colors,
        binary=True,
        fig_ax=(fig, axs[2]),
        )

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=qs[:, :, 0],
            timeseries_name='q',
            color=colors,
            binary=True,
            fig_ax=(fig, axs[3]),
            )

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=hs[:, :, 0],
            timeseries_name='a',
            color=colors,
            binary=True,
            fig_ax=(fig, axs[4]),
            )

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=us[:, :, 0],
            timeseries_name='u',
            color=colors,
            binary=True,
            fig_ax=(fig, axs[5]),
        )
        
        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=bs[:, :, 0],
            timeseries_name='b',
            color=colors,
            binary=True,
            fig_ax=(fig, axs[6]),
        )

        plt.show()

    feature_list = ['beta']
    coefficient_list = [beta]
    for m in sindy_models:
        feature_list += sindy_models[m].get_feature_names()
        coefficient_list += [c for c in sindy_models[m].coefficients()[0]]
    
    # remove features and coefficients with the name 'u'
    idx_u = [i for i, f in enumerate(feature_list) if 'u' == f]
    popped = 0
    for i in idx_u:
        feature_list.pop(i-popped)
        coefficient_list.pop(i-popped)
        popped += 1
    
    return tuple(feature_list), tuple(coefficient_list)


if __name__=='__main__':
    main(
        # model = 'params/dataset_ensemble_noise_analysis/params_rnn_sessions4096_submodels16_noise3',
        
        # sindy parameters
        polynomial_degree=2,
        threshold=0.05,
        regularization=0,
        
        # generated training dataset parameters
        n_trials_per_session = 200,
        n_sessions = 100,
        
        # rnn parameters
        hidden_size = 8,
        
        # ground truth parameters
        alpha = 0.25,
        beta = 3,
        forget_rate = 0.,
        perseveration_bias = 0.25,
        regret = False,
        confirmation_bias = False,
        # reward_update_rule = lambda q, reward: reward-q,
        
        # environment parameters
        sigma = 0.1,
        
        analysis=True,
    )