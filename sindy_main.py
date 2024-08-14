import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

import pysindy as ps

sys.path.append('resources')
from resources.rnn import RLRNN, EnsembleRNN
from resources.bandits import AgentQ, AgentNetwork, EnvironmentBanditsDrift, plot_session, get_update_dynamics, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, check_library_setup, constructor_update_rule_sindy, sindy_loss_x, sindy_loss_z
from resources.rnn_utils import parameter_file_naming
from resources.sindy_training import fit_model, setup_sindy_agent

warnings.filterwarnings("ignore")

def main(
    # sindy parameters
    threshold = 0.03,
    polynomial_degree = 1,
    regularization = 1e-2,
    sindy_ensemble = False,
    library_ensemble = False,
    
    # ground truth parameters
    alpha = 0.25,
    beta = 3,
    forget_rate = 0.,
    perseveration_bias = 0.,
    correlated_update = False,
    regret = False,
    
    # environment parameters
    sigma = 0.2,
    non_binary_reward = False,
    
    analysis=False,
    ):

    # training dataset parameters
    n_trials_per_session = 200
    n_sessions = 10

    # environment parameters
    n_actions = 2
    sigma = .2
    non_binary_reward = False
    correlated_reward = False

    # rnn parameters
    hidden_size = 4
    last_output = False
    last_state = False
    use_lstm = False
    voting_type = EnsembleRNN.MEDIAN

    # tracked variables in the RNN
    z_train_list = ['xQf','xQr_r', 'xQr_p', 'xH']
    control_list = ['ca', 'cr', 'cdQr[k-1]', 'cdQr[k-2]']
    sindy_feature_list = z_train_list + control_list

    # library setup aka which terms are allowed as control inputs in each SINDy model
    # key is the SINDy submodel name, value is a list of allowed control inputs
    library_setup = {
        'xQf': [],
        'xQr_r': [],#['cdQr[k-2]', 'cdQr[k-1]'],
        'xQr_p': [],#['cdQr[k-2]', 'cdQr[k-1]'],
        'xH': []
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
        'xQf': ['ca', 0, True],
        'xQr_r': [['ca', 1, True], ['cr', 1, False]],
        'xQr_p': [['ca', 1, True], ['cr', 0, False]],
        'xH': ['ca', 1, True]
    }

    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')

    # set up ground truth agent and environment
    environment = EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_reward=non_binary_reward, correlated_reward=correlated_reward)
    agent = AgentQ(alpha, beta, n_actions, forget_rate, perseveration_bias, correlated_update)
    dataset_test, experiment_list_test = create_dataset_bandits(agent, environment, 200, 1)

    # set up rnn agent and expose q-values to train sindy
    params_path = parameter_file_naming('params/params', use_lstm, alpha, beta, forget_rate, perseveration_bias, correlated_update, regret, non_binary_reward, verbose=True)
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))['model']
    rnn = RLRNN(n_actions, hidden_size, 0.5, last_output, last_state, sindy_feature_list)
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

    # create dataset for sindy training, fit sindy, set up sindy agent
    z_train, control, feature_names, beta = create_dataset(agent_rnn, environment, n_trials_per_session, n_sessions, normalize=True, shuffle=False, trimming=100)
    sindy_models = fit_model(z_train, control, feature_names, polynomial_degree, library_setup, datafilter_setup, True, False, threshold, regularization)
    update_rule_sindy = constructor_update_rule_sindy(sindy_models)
    agent_sindy = setup_sindy_agent(update_rule_sindy, n_actions, False, experiment_list_test[0], agent_rnn, True)
    print(f'\nBeta for SINDy: {beta}')
    agent_sindy._beta = beta
    # loss_x = sindy_loss_x(agent_sindy, dataset_test)
    # loss_z = sindy_loss_z(agent_sindy, dataset_test, agent_rnn)
    # print(f'\nLoss for SINDy in x-coordinates: {np.round(loss_x, 4)}')
    # print(f'Loss for SINDy in z-coordinates: {np.round(loss_z, 4)}')
    # dataset_sindy, experiment_list_sindy = create_dataset_bandits(agent_sindy, environment, n_trials_per_session, 1)

    # --------------------------------------------------------------
    # Analysis
    # --------------------------------------------------------------

    if analysis:
        labels = ['Ground Truth', 'RNN', 'SINDy']
        experiment_test = experiment_list_test[0]
        choices = experiment_test.choices
        rewards = experiment_test.rewards

        list_probs = []
        list_qs = []

        # get q-values from groundtruth
        qs_test, probs_test = get_update_dynamics(experiment_test, agent)
        list_probs.append(np.expand_dims(probs_test, 0))
        list_qs.append(np.expand_dims(qs_test, 0))

        # get q-values from trained rnn
        qs_rnn, probs_rnn = get_update_dynamics(experiment_test, agent_rnn)
        list_probs.append(np.expand_dims(probs_rnn, 0))
        list_qs.append(np.expand_dims(qs_rnn, 0))

        # get q-values from trained sindy
        qs_sindy, probs_sindy = get_update_dynamics(experiment_test, agent_sindy)
        list_probs.append(np.expand_dims(probs_sindy, 0))
        list_qs.append(np.expand_dims(qs_sindy, 0))

        colors = ['tab:blue', 'tab:orange', 'tab:pink', 'tab:grey']

        # concatenate all choice probs and q-values
        probs = np.concatenate(list_probs, axis=0)
        qs = np.concatenate(list_qs, axis=0)

        # normalize q-values
        def normalize(qs):
            return (qs - np.min(qs, axis=1, keepdims=True)) / (np.max(qs, axis=1, keepdims=True) - np.min(qs, axis=1, keepdims=True))

        qs = normalize(qs)

        fig, axs = plt.subplots(4, 1, figsize=(20, 10))
        # turn the x labels off for all but the last subplot
        for i in range(4):
            axs[i].set_xticklabels([])
            axs[i].set_xlabel('')
            axs[i].set_xlim(0, 200)
            # axs[i].set_ylim(0, 1)    

        reward_probs = np.stack([experiment_test.timeseries[:, i] for i in range(n_actions)], axis=0)
        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=reward_probs,
            timeseries_name='Reward Probs',
            labels=[f'Arm {a}' for a in range(n_actions)],
            color=['tab:purple', 'tab:cyan'],
            binary=not non_binary_reward,
            fig_ax=(fig, axs[0]),
            x_label='',
            )

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=probs[:, :, 0],
            timeseries_name='Choice Probs',
            color=colors,
            labels=labels,
            binary=not non_binary_reward,
            fig_ax=(fig, axs[1]),
            x_label='',
            )

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=qs[:, :, 0],
            timeseries_name='Q Arm 0',
            color=colors,
            binary=not non_binary_reward,
            fig_ax=(fig, axs[2]),
            x_label='',
            )

        # plot_session(
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

        plot_session(
            compare=True,
            choices=choices,
            rewards=rewards,
            timeseries=dqs_t[:, :, 0],
            timeseries_name='dQ/dt',
            color=colors,
            binary=not non_binary_reward,
            fig_ax=(fig, axs[3]),
            )

        # dqs_arms = -1*np.diff(qs, axis=2)
        # dqs_arms = normalize(dqs_arms)

        # plot_session(
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
        # ground truth parameters
        alpha = 0.25,
        beta = 3,
        forget_rate = 0.,
        perseveration_bias = 0.,
        regret = False,
        
        # environment parameters
        sigma = 0.2,
        non_binary_reward = False,
        
        # auxiliary
        analysis=True,
    )