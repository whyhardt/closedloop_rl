import sys
import warnings

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

sys.path.append('resources')
from resources.rnn import RLRNN
from resources.bandits import AgentQ, AgentSindy, AgentNetwork, EnvironmentBanditsDrift, EnvironmentBanditsSwitch, plot_session, create_dataset as create_dataset_bandits
from resources.sindy_utils import create_dataset, check_library_setup
from resources.rnn_utils import parameter_file_naming
from resources.sindy_training import fit_model
from utils.convert_dataset import convert_dataset
from utils.plotting import session

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
    
    # ground truth parameters
    beta_reward = 3.,
    alpha = 0.25,
    alpha_penalty = -1.,
    alpha_counterfactual = 0.,
    beta_choice = 0.,
    alpha_choice = 0.,
    forget_rate = 0.,
    confirmation_bias = 0.,
    parameter_variance = 0.,
    reward_prediction_error: Callable = None,
    
    # environment parameters
    n_actions = 2,
    sigma = .1,
    
    analysis: bool = False, 
    ):

    # tracked variables in the RNN
    x_train_list = ['x_V_LR', 'x_V_LR_cf', 'x_V_nc', 'x_C', 'x_C_nc']
    control_list = ['c_a', 'c_r', 'c_r_cf', 'c_a_repeat', 'c_V']
    sindy_feature_list = x_train_list + control_list

    # library setup aka which terms are allowed as control inputs in each SINDy model
    # key is the SINDy submodel name, value is a list of allowed control inputs
    library_setup = {
        'x_V_LR': ['c_V', 'c_r'],  # learning rate for chosen action
        'x_V_LR_cf': ['c_V', 'c_r_cf'],  # learning rate for not-chosen action in counterfactual setup -> same inputs as for chosen action because inputs are 2D (action_0, action_1) -> difference made in filter setup!
        'x_V_nc': [],
        'x_C': ['c_a_repeat'],
        'x_C_nc': [],
    }

    # data-filter setup aka which samples are allowed as training samples in each SINDy model based on the given filter condition
    # key is the SINDy submodel name, value is a list with a triplet of values: 
    #   1. str: feature name to be used as a filter
    #   2. numeric: the numeric filter condition
    #   3. bool: remove feature from control inputs
    # Can also be a list of list of triplets for multiple filter conditions
    # Example:
    # 'x_V_nc': ['c_a', 0, True] means that only samples where the feature 'c_a' is 0 are used for training the SINDy model 'x_V_nc' and the control parameter 'c_a' is removed for training the model
    datafilter_setup = {
        'x_V_LR': ['c_a', 1, True],  # learning rate for chosen action
        'x_V_LR_cf': ['c_a', 0, True],  # learning rate for not-chosen action in counterfactual setup
        'x_V_nc': ['c_a', 0, True],
        'x_C': ['c_a', 1, True],
        'x_C_nc': ['c_a', 0, True],
    }

    if not check_library_setup(library_setup, sindy_feature_list, verbose=True):
        raise ValueError('Library setup does not match feature list.')
    
    agent = None
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
        _, experiment_list_train, df, _ = convert_dataset(data)
        experiment_list_test = None
        
        if session_id is None:
            session_id = np.arange(len(experiment_list_train))
        else:
            session_id = [session_id]
        
        if analysis and 'mean_beta_reward' in df.columns:
            # get parameters from dataset
            agent = AgentQ(
                beta_reward = df['beta_reward'].values[(df['session']==session_id[0]).values][0],
                alpha_reward = df['alpha_reward'].values[(df['session']==session_id[0]).values][0],
                alpha_penalty = df['alpha_penalty'].values[(df['session']==session_id[0]).values][0],
                confirmation_bias = df['confirmation_bias'].values[(df['session']==session_id[0]).values][0],
                forget_rate = df['forget_rate'].values[(df['session']==session_id[0]).values][0],
                beta_choice = df['beta_choice'].values[(df['session']==session_id[0]).values][0],
                alpha_choice = df['alpha_choice'].values[(df['session']==session_id[0]).values][0],
            )

    # set up rnn agent and expose q-values to train sindy
    if model is None:
        params_path = parameter_file_naming('params/params', beta_reward=beta_reward, alpha_reward=alpha, alpha_penalty=alpha_penalty, beta_choice=beta_choice, alpha_choice=alpha_choice, forget_rate=forget_rate, confirmation_bias=confirmation_bias, alpha_counterfactual=alpha_counterfactual, variance=parameter_variance, verbose=True)
    else:
        params_path = model
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))['model']
    counterfactual = np.mean(experiment_list_train[0].rewards[:, -1]) != -1
    participant_embedding = 'participant_embedding.weight' in state_dict.keys()
    key_hidden_size = [key for key in state_dict if 'x' in key.lower()][0]  # first key that contains the hidden_size
    hidden_size = state_dict[key_hidden_size].shape[0]
    rnn = RLRNN(
        n_actions=n_actions, 
        hidden_size=hidden_size,
        n_participants=len(experiment_list_train) if participant_embedding else 0, 
        init_value=0.5, 
        list_sindy_signals=sindy_feature_list, 
        counterfactual=counterfactual,
        )
    print('Loaded model ' + params_path)
    rnn.load_state_dict(state_dict)
    agent_rnn = AgentNetwork(rnn, n_actions, deterministic=True)            

    agent_sindy = {}
    for id in session_id:
        # get SINDy-formatted data with exposed latent variables computed by RNN-Agent
        x_train, control, feature_names = create_dataset(
            agent=agent_rnn, 
            data=[experiment_list_train[id]],
            n_trials_per_session=n_trials_per_session, 
            n_sessions=n_sessions, 
            shuffle=False,
            trimming=True,
            clear_offset=True,
            )
        
        # fit SINDy models -> One model per x_train feature
        sindy_models = fit_model(
            x_train=x_train, 
            control=control, 
            feature_names=feature_names, 
            polynomial_degree=polynomial_degree, 
            library_setup=library_setup, 
            filter_setup=datafilter_setup, 
            verbose=verbose, 
            get_loss=False, 
            optimizer_threshold=threshold, 
            optimizer_alpha=regularization,
            )
        
        # set up SINDy-Agent -> One SINDy-Agent per session if participant embedding is activated
        agent_rnn.new_sess(experiment_list_train[id].session[0])
        agent_sindy[id] = AgentSindy(
            sindy_models=sindy_models, 
            n_actions=n_actions, 
            beta_reward=agent_rnn._beta_reward,
            beta_choice=agent_rnn._beta_choice, 
            deterministic=True, 
            counterfactual=agent_rnn._model._counterfactual,
            )
    
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
        
        if experiment_list_test is None:
            experiment_list_test = [experiment_list_train[session_id[0]]]
        
        # print sindy equations from tested sindy agent
        agent_sindy[session_id[0]].new_sess()
        print('\nDiscovered SINDy models:')
        for model in agent_sindy[session_id[0]]._models:
            agent_sindy[session_id[0]]._models[model].print()
        print(f'(x_beta_r) = {agent_sindy[session_id[0]]._beta_reward:.3f}')
        print(f'(x_beta_c) = {agent_sindy[session_id[0]]._beta_choice:.3f}')
        print('\n')
        
        # get analysis plot
        if agent is not None:
            agents = {'groundtruth': agent, 'rnn': agent_rnn, 'sindy': agent_sindy[session_id[0]]}
            plt_title = '$True:\\beta_r=$'+str(np.round(agent._beta_reward, 2)) + '; $\\beta_c=$'+str(np.round(agent._beta_choice, 2))+'\n'
        else:
            agents = {'rnn': agent_rnn, 'sindy': agent_sindy[session_id[0]]}
            plt_title = ''
            
        experiment_analysis = experiment_list_test[0]
        fig, axs = session(agents, experiment_analysis)
        plt_title += '$Sim:\\beta_r=$'+str(np.round(agent_sindy[session_id[0]]._beta_reward, 2)) + '; $\\beta_c=$'+str(np.round(agent_sindy[session_id[0]]._beta_choice, 2))
        
        fig.suptitle(plt_title)
        plt.show()
        
    # save a dictionary of trained features per model
    features = {
        'beta_reward': (('beta_reward'), (agent_sindy[session_id[0]]._beta_reward)),
        'beta_choice': (('beta_choice'), (agent_sindy[session_id[0]]._beta_choice)),
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
    
    for id in agent_sindy:
        agent_sindy[id].new_sess()
        
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