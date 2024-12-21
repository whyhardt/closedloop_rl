import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Callable

sys.path.append('resources')
from resources.rnn import RLRNN
from resources.bandits import AgentQ, AgentSindy, AgentNetwork, EnvironmentBanditsDrift, EnvironmentBanditsSwitch, plot_session, get_update_dynamics, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, check_library_setup, bandit_loss
from resources.rnn_utils import parameter_file_naming
from resources.sindy_training import fit_model
from utils.convert_dataset import convert_dataset
from utils.plotting import plot_session

warnings.filterwarnings("ignore")

def main(
    model: str = None,
    data: str = None,
    
    # generated training dataset parameters
    n_trials_per_session = 256,
    n_sessions = 1,
    session_id: int = None,
    
    # sindy parameters
    threshold = 0.03,
    polynomial_degree = 1,
    regularization = 1e-2,
    verbose = True,
    
    # rnn parameters
    hidden_size = 8,
    
    # ground truth parameters
    beta_reward = 3.,
    alpha = 0.25,
    alpha_penalty = -1.,
    alpha_counterfactual = 0.,
    beta_choice = 0.,
    alpha_choice = 1.,
    forget_rate = 0.,
    confirmation_bias = 0.,
    reward_prediction_error: Callable = None,
    
    # environment parameters
    n_actions = 2,
    sigma = .1,
    
    analysis: bool = False,
    ):

    # tracked variables in the RNN
    z_train_list = ['xLR', 'xLR_cf', 'xQf', 'xC', 'xCf']
    control_list = ['ca', 'cr', 'cp', 'cr_cf', 'cp_cf', 'ca_repeat', 'cQ']
    sindy_feature_list = z_train_list + control_list

    # library setup aka which terms are allowed as control inputs in each SINDy model
    # key is the SINDy submodel name, value is a list of allowed control inputs
    library_setup = {
        'xLR': ['cQ', 'cr', 'cp'],  # learning rate for chosen action
        'xLR_cf': ['cQ', 'cr', 'cp'],  # learning rate for not-chosen action in counterfactual setup -> same inputs as for chosen action because inputs are 2D (action_0, action_1) -> difference made in filter setup!
        'xQf': [],
        'xC': ['ca_repeat'],
        'xCf': [],
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
        'xLR': ['ca', 1, True],  # learning rate for chosen action
        'xLR_cf': ['ca', 0, True],  # learning rate for not-chosen action in counterfactual setup
        'xQf': ['ca', 0, True],
        'xC': ['ca', 1, True],
        'xCf': ['ca', 0, True],
    }

    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')

    if data is None:
        # set up ground truth agent and environment
        environment = EnvironmentBanditsDrift(sigma, n_actions)
        # environment = EnvironmentBanditsSwitch(sigma)
        agent = AgentQ(n_actions, beta_reward=beta_reward, alpha_reward=alpha, alpha_penalty=alpha_penalty, beta_choice=beta_choice, alpha_choice=alpha_choice, forget_rate=forget_rate, confirmation_bias=confirmation_bias, alpha_counterfactual=alpha_counterfactual)
        if reward_prediction_error is not None:
            agent.set_reward_prediction_error(reward_prediction_error)
        _, experiment_list_test, _ = create_dataset_bandits(agent, environment, 100, 1)
        _, experiment_list_train, _ = create_dataset_bandits(agent, environment, n_trials_per_session, n_sessions)
    else:
        # get data from experiments
        _, experiment_list_train, _, _ = convert_dataset(data)
        index_test = len(experiment_list_train)-1
        experiment_list_test = [experiment_list_train[index_test]]
        # experiment_list_train = experiment_list_train[:-1]

    # set up rnn agent and expose q-values to train sindy
    if model is None:
        params_path = parameter_file_naming('params/params', beta_reward=beta_reward, alpha=alpha, alpha_penalty=alpha_penalty, beta_choice=beta_choice, alpha_choice=alpha_choice, forget_rate=forget_rate, confirmation_bias=confirmation_bias, alpha_counterfactual=alpha_counterfactual, variance=parameter_variance, verbose=True)
    else:
        params_path = model
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))['model']
    counterfactual = np.mean(experiment_list_train[0].rewards[:, -1]) != -1
    participant_emb = 'participant_embedding.weight' in state_dict.keys()
    rnn = RLRNN(n_actions=n_actions, hidden_size=hidden_size, n_participants=len(experiment_list_train), init_value=0.5, list_sindy_signals=sindy_feature_list, participant_emb=participant_emb, counterfactual=counterfactual)
    print('Loaded model ' + params_path)
    rnn.load_state_dict(state_dict)
    agent_rnn = AgentNetwork(rnn, n_actions, deterministic=True)            

    # create dataset for sindy training, fit sindy, set up sindy agent
    if session_id is not None:
        experiment_list_train = [experiment_list_train[session_id]]
        index_test = 0
    agent_sindy = []
    for i in range(len(experiment_list_train)):
        z_train, control, feature_names = create_dataset(agent_rnn, [experiment_list_train[i]], n_trials_per_session, n_sessions, clear_offset=False, shuffle=False, trimming=True)
        sindy_models = fit_model(z_train, control, feature_names, polynomial_degree, library_setup, datafilter_setup, verbose, False, threshold, regularization)
        agent_rnn.new_sess(experiment_list_train[i].session[0])
        agent_sindy.append(AgentSindy(sindy_models, n_actions, (agent_rnn._beta_reward, agent_rnn._beta_choice), True, agent_rnn._model._counterfactual))
    
    # if verbose:
    #     print(f'SINDy Beta: {agent_rnn._model._beta_reward.item():.2f} and {agent_rnn._model._beta_choice.item():.2f}')
    #     print('Calculating RNN and SINDy loss in X (predicting behavior; Target: Subject)...', end='\r')
    #     test_loss_rnn_x = bandit_loss(agent_rnn, experiment_list_test, coordinates="x")
    #     test_loss_sindy_x = bandit_loss(agent_sindy, experiment_list_test, coordinates="x")
    #     print(f'RNN Loss in X: {test_loss_rnn_x}')
    #     print(f'SINDy Loss in X: {test_loss_sindy_x}')

    # --------------------------------------------------------------
    # Analysis
    # --------------------------------------------------------------

    if analysis:
        
        # print sindy equations from tested sindy agent
        agent_sindy[index_test].new_sess()
        for model in agent_sindy[index_test]._models:
            agent_sindy[index_test]._models[model].print()
        
        # get analysis plot
        if data is None:
            agents = {'groundtruth': agent, 'rnn': agent_rnn, 'sindy': agent_sindy[index_test]}
        else:
            agents = {'rnn': agent_rnn, 'sindy': agent_sindy[index_test]}
        experiment_analysis = experiment_list_test[0]
        fig, axs = plot_session(agents, experiment_analysis)
        fig.suptitle('$beta_r=$'+str(np.round(agent_rnn._beta_reward, 2)) + '; $beta_c=$'+str(np.round(agent_rnn._beta_choice, 2)))
        plt.show()
        
    # save a dictionary of trained features per model
    features = {
        'beta_reward': (('beta_reward'), (agent_sindy[index_test]._beta_reward)),
        'beta_choice': (('beta_choice'), (agent_sindy[index_test]._beta_choice)),
        }
    for m in sindy_models:
        features[m] = []
        features_i = sindy_models[m].get_feature_names()
        coeffs_i = [c for c in sindy_models[m].coefficients()[0]]
        index_u = []
        for i, f in enumerate(features_i):
            if 'u' in f:
                index_u.append(i)
        features_i = [item for idx, item in enumerate(features_i) if idx not in index_u]
        coeffs_i = [item for idx, item in enumerate(coeffs_i) if idx not in index_u]
        features[m].append(tuple(features_i))
        features[m].append(tuple(coeffs_i))
        features[m] = tuple(features[m])
    
    for i in range(len(agent_sindy)):
        agent_sindy[i].new_sess()
        
    return agent_sindy, sindy_models, features


if __name__=='__main__':
    main(
        model = 'params/benchmarking/rnn_sugawara.pkl',
        data = 'data/2arm/sugawara2021_143_processed.csv',
        n_trials_per_session=None,
        n_sessions=None,
        verbose=False,
        
        # sindy parameters
        polynomial_degree=2,
        threshold=0.05,
        regularization=0,
        
        # rnn parameters
        hidden_size = 8,
        
        # generated training dataset parameters
        # n_trials_per_session = 200,
        # n_sessions = 100,
        
        # ground truth parameters
        # alpha = 0.25,
        # beta = 3,
        # forget_rate = 0.,
        # perseverance_bias = 0.25,
        # alpha_penalty = 0.5,
        # confirmation_bias = 0.5,
        # reward_update_rule = lambda q, reward: reward-q,
        
        # environment parameters
        # sigma = 0.1,
        
        analysis=True,
    )