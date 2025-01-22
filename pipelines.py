import sys
import warnings
import os

# warnings.filterwarnings("ignore")

import time
import torch
import numpy as np
from copy import deepcopy

sys.path.append('resources')
from resources.rnn import RLRNN
from resources.bandits import AgentSindy, AgentNetwork, get_update_dynamics
from resources.sindy_utils import create_dataset, check_library_setup
from resources.sindy_training import fit_model
from resources.rnn_utils import DatasetRNN, load_checkpoint
from utils.convert_dataset import convert_dataset

from theorist import rl_sindy_theorist

warnings.filterwarnings("ignore")


def pipeline_rnn(
    data: str,

    # rnn parameters
    model_checkpoint: str = None,
    hidden_size = 8,
    dropout = 0.5,
    participant_emb = False,

    # data and training parameters
    n_actions = 2,
    epochs = 128,
    train_test_ratio = 0.7,
    bagging = False,
    sequence_length = None,
    n_steps_per_call = 16,  # -1 for full sequence
    batch_size = -1,  # -1 for one batch per epoch
    learning_rate = 1e-2,
    convergence_threshold = 1e-6,
    
    verbose = False,
    ):
  
    if not os.path.exists('params'):
        os.makedirs('params')

    # tracked variables in the RNN
    x_train_list = ['x_V_LR', 'x_V_LR_cf', 'x_V_nc', 'x_C', 'x_C_nc']
    control_list = ['c_a', 'c_r', 'c_r_cf', 'c_a_repeat', 'c_V']
    sindy_feature_list = x_train_list + control_list

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = convert_dataset(data, sequence_length=sequence_length)[0]
    n_participants = dataset.xs.shape[0]

    if train_test_ratio < 1 and train_test_ratio >= 0:
        # setup of training and test dataset
        index_train = int((1-train_test_ratio) * dataset.xs.shape[1])
    elif train_test_ratio < 0:
        raise ValueError(f'Argument train_test_ratio ({train_test_ratio}) cannot be lower than 0.')

    xs_test, ys_test = dataset.xs[:, index_train:], dataset.ys[:, index_train:]
    xs_train, ys_train = dataset.xs[:, :index_train], dataset.ys[:, :index_train]
    dataset_train = DatasetRNN(xs_train, ys_train, sequence_length=sequence_length)
    dataset_test = DatasetRNN(xs_test, ys_test)

    # define model
    model_rnn = RLRNN(
        n_actions=n_actions, 
        hidden_size=hidden_size, 
        init_value=0.5,
        device=device,
        list_sindy_signals=sindy_feature_list,
        dropout=dropout,
        n_participants=n_participants if participant_emb else 0,
        counterfactual=dataset_train.xs[:, :, n_actions+1].mean() != -1,
        ).to(device)

    optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=learning_rate)

    if verbose:
        print('Setup of the RNN model complete.')

    if model_checkpoint is not None:
        model_rnn, optimizer_rnn = load_checkpoint(model_checkpoint, model_rnn, optimizer_rnn)
        if verbose:
            print('Loaded model parameters.')

    loss_test = None
    if epochs > 0:
        start_time = time.time()

    #Fit the RNN
    if verbose:
        print('Training the RNN...')
    model_rnn, optimizer_rnn, loss_train = fit_model(
        model=model_rnn,
        dataset_train=dataset_train,
        optimizer=optimizer_rnn,
        convergence_threshold=convergence_threshold,
        epochs=epochs,
        batch_size=batch_size,
        bagging=bagging,
        n_steps_per_call=n_steps_per_call,
    )

    model_rnn.eval()
    
    # save trained parameters
    state_dict = {'model': model_rnn.state_dict(), 'optimizer': optimizer_rnn.state_dict()}
    torch.save(state_dict, model_checkpoint)
    
    if verbose:
        print('Training finished.')
        print(f'Saved RNN parameters to file {model_checkpoint}.')
        print(f'Training took {time.time() - start_time:.2f} seconds.')
    
    # validate model
    loss_test = None
    if dataset_test is not None:
        if verbose:
            print('\nTesting the trained RNN on the test dataset...')
        with torch.no_grad():
            _, _, loss_test = fit_model(
                model=model_rnn,
                dataset_train=dataset_train,
            )

    loss = loss_test if loss_test is not None else loss_train
    
    return model_rnn, loss


def pipeline_sindy(
    model_rnn: RLRNN,
    data: str,
    
    # environment parameters
    n_actions: int = 2,
    
    # sindy parameters
    threshold = 0.03,
    polynomial_degree = 1,
    regularization = 1e-2,
    verbose = False, 
    ):

    # -------------------------------------------------------------------
    # SINDY SETUP
    # -------------------------------------------------------------------
    
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
    
    # -------------------------------------------------------------------
    # SINDY TRAINING
    # -------------------------------------------------------------------
    
    # get data from experiments
    experiment_list_train = convert_dataset(data)[1]
    
    # set sessions ids according to whether the rnn fitted single participants (i.e. with participant embedding)
    session_id = np.arange(len(experiment_list_train)) if model_rnn._n_participants > 0 else [0]
    
    # set up rnn agent and expose q-values to train sindy
    agent_rnn = AgentNetwork(model_rnn, n_actions, deterministic=True)            

    agent_sindy = {}
    features = {}
    for id in session_id:
        # get SINDy-formatted data with exposed latent variables computed by RNN-Agent
        x_train, control, feature_names = create_dataset(
            agent=agent_rnn, 
            data=[experiment_list_train[id]] if len(session_id)>1 else experiment_list_train,
            n_trials_per_session=None, 
            n_sessions=None, 
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
        
        # test sindy agent and see if it breaks
        try:
            # set up SINDy-Agent
            agent_rnn.new_sess(experiment_list_train[id].session[0])
            agent_sindy[id] = AgentSindy(
                sindy_models=sindy_models, 
                n_actions=n_actions, 
                beta_reward=agent_rnn._beta_reward,
                beta_choice=agent_rnn._beta_choice, 
                deterministic=True, 
                counterfactual=agent_rnn._model._counterfactual,
                )
            
            get_update_dynamics(experiment_list_train[id], agent_sindy[id])
            
            agent_sindy[id].new_sess()
    
            # save a dictionary of trained features per model
            features_id = {
                'beta_reward': (('beta_reward'), (agent_sindy[id]._beta_reward)),
                'beta_choice': (('beta_choice'), (agent_sindy[id]._beta_choice)),
                }
            
            for m in sindy_models:
                features_m = sindy_models[m].get_feature_names()
                coeffs_m = [c for c in sindy_models[m].coefficients()[0]]
                # sort out every dummy control parameter (i.e. any candidate term which contains 'u')
                index_u = []
                for i, f in enumerate(features_m):
                    if 'u' in f:
                        index_u.append(i)
                features_m = [item for idx, item in enumerate(features_m) if idx not in index_u]
                coeffs_m = [item for idx, item in enumerate(coeffs_m) if idx not in index_u]
                features_id[m] = (tuple(features_m), tuple(coeffs_m))
            
            features[id] = deepcopy(features_id)
        except Exception as e:
            for m in sindy_models:
                features_id[m] = (str(e), str(e))
            features[id] = deepcopy(features_id)

    return features