import sys
import warnings

import time
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.base import BaseEstimator
from typing import Dict

sys.path.append('resources')
from resources.rnn import BaseRNN
from resources.bandits import AgentSindy, AgentNetwork, get_update_dynamics, BanditSession
from resources.sindy_utils import create_dataset, check_library_setup
from resources.sindy_training import fit_model as fit_sindy
from resources.rnn_utils import DictDataset
from resources.rnn_training import fit_model as fit_rnn

warnings.filterwarnings("ignore")


class rl_sindy_theorist(BaseEstimator):
    
    def __init__(
        self,
        
        # rnn parameters
        hidden_size = 8,
        dropout = 0.5,

        # data parameters
        n_actions = 2,
        n_participants = 1,
        counterfactual = False,
        
        # rnn training parameters
        epochs = 128,
        bagging = False,
        sequence_length = None,
        n_steps_per_call = 16,  # -1 for full sequence
        batch_size = -1,  # -1 for one batch per epoch
        learning_rate = 1e-2,
        convergence_threshold = 1e-6,
        device = torch.device('cpu'),
        
        # sindy parameters
        threshold = 0.03,
        polynomial_degree = 2,
        regularization = 1e-2,
        
        verbose = False,
        ):
        
        super(BaseEstimator, self).__init__()
        
        # -------------------------------------------------------------------
        # TRAINING PARAMETERS
        # -------------------------------------------------------------------
        
        self.epochs = epochs
        self.bagging = bagging
        self.sequence_length = sequence_length
        self.n_steps_per_call = n_steps_per_call  # -1 for full sequence
        self.batch_size = batch_size  # -1 for one batch per epoch
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold
        self.device = device
        self.verbose = verbose
        
        self.threshold = threshold
        self.polynomial_degree = polynomial_degree
        self.regularization = regularization
        
        self.n_actions = n_actions
        self.n_participants = n_participants
        
        # -------------------------------------------------------------------
        # NETWORK SETUP
        # -------------------------------------------------------------------
        
        # tracked variables in the RNN
        x_train_list = ['x_ValueRewardCH']
        control_list = ['c_Action', 'c_Reward', 'c_ParticipantID']
        self.sindy_feature_list = x_train_list + control_list

        # library setup aka which terms are allowed as control inputs in each SINDy model
        # key is the SINDy submodel name, value is a list of allowed control inputs
        self.module_inputs = {
            'x_ValueRewardCH': ['c_Reward'],
        }

        # data-filter setup aka which samples are allowed as training samples in each SINDy model based on the given filter condition
        # key is the SINDy submodel name, value is a list with a triplet of values: 
        #   1. str: feature name to be used as a filter
        #   2. numeric: the numeric filter condition
        #   3. bool: remove feature from control inputs
        # Can also be a list of list of triplets for multiple filter conditions
        # Example:
        # 'x_V_nc': ['c_a', 0, True] means that only samples where the feature 'c_a' is 0 are used for training the SINDy model 'x_V_nc' and the control parameter 'c_a' is removed for training the model
        self.module_filter = {
            'x_ValueRewardCH': ['c_Action', 1, True],
        }

        if not check_library_setup(self.module_inputs, self.sindy_feature_list, verbose=True):
            raise ValueError('\nLibrary setup does not match feature list.')
        
        self.sindy_agents = None
        self.sindy_features = None
        
        # -------------------------------------------------------------------
        # RNN SETUP
        # -------------------------------------------------------------------
        
        self.device = device
        
        self.model_rnn = BaseRNN(
            n_actions=n_actions, 
            hidden_size=hidden_size, 
            signals=x_train_list+control_list,
            dropout=dropout,
            n_participants=n_participants,
            module_inputs=self.module_inputs,
            module_filter=self.module_filter,
            ).to(device)
        
        self.optimizer_rnn = torch.optim.Adam(self.model_rnn.parameters(), lr=learning_rate)
        
    
    def fit(self, conditions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        
        dataset = DictDataset(conditions, targets)
        start_time = time.time()
        
        # -------------------------------------------------------------------
        # FIT RNN
        # -------------------------------------------------------------------

        batch_size = conditions[tuple(conditions.keys())[0]].shape[0] if self.batch_size == -1 else self.batch_size
        
        if self.verbose:
            print('\nTraining the RNN...')
        model_rnn, optimizer_rnn, _ = fit_rnn(
            model=self.model_rnn,
            dataset_train=dataset,
            optimizer=self.optimizer_rnn,
            convergence_threshold=self.convergence_threshold,
            epochs=self.epochs,
            batch_size=batch_size,
            bagging=self.bagging,
            n_steps_per_call=self.n_steps_per_call,
        )

        model_rnn.eval()
        self.model_rnn = model_rnn
        self.optimizer_rnn = optimizer_rnn
        
        if self.verbose:
            print('\nTraining finished.')
            print(f'Training took {time.time() - start_time:.2f} seconds.')
            
        # -------------------------------------------------------------------
        # FIT SINDY
        # -------------------------------------------------------------------
        
        self.sindy_agents = {}
        self.sindy_features = {}
        
        # transform DatasetRNN to List[BanditSession]
        experiment_list = []        
        for index_session in range(len(dataset)):
            n_trials = np.argmin(np.sum(dataset.xs['c_Reward'][index_session].detach().cpu().numpy(), axis=-1))
            experiment_list.append(BanditSession(
                choices=dataset.xs['c_Action'][index_session, :n_trials].detach().cpu().numpy(),
                rewards=dataset.xs['c_Reward'][index_session, :n_trials].detach().cpu().numpy(),
                session=dataset.xs['c_ParticipantID'][index_session, :n_trials].detach().cpu().numpy(),
                reward_probabilities=np.full_like(dataset.xs['c_Action'][index_session, :n_trials], -1),
                q = np.full_like(dataset.xs['c_Action'][index_session, :n_trials], -1),
                n_trials = n_trials,
            ))
        
        # set sessions ids according to whether the rnn fitted single participants (i.e. with participant embedding)
        session_id = np.arange(len(experiment_list)) if self.n_participants > 1 else [0]
        
        # set up rnn agent and expose q-values to train sindy
        agent_rnn = AgentNetwork(model_rnn, self.n_actions, deterministic=True, device=self.device)
        
        for id in session_id:
            # get SINDy-formatted data with exposed latent variables computed by RNN-Agent
            x_train, control, feature_names = create_dataset(
                agent=agent_rnn, 
                data=[experiment_list[id]] if len(session_id)>1 else experiment_list,
                n_trials_per_session=None,
                n_sessions=None,
                shuffle=False,
                trimming=True,
                clear_offset=True,
                )
            
            # fit SINDy models -> One model per x_train feature
            sindy_models = fit_sindy(
                x_train=x_train, 
                control=control, 
                feature_names=feature_names, 
                polynomial_degree=self.polynomial_degree, 
                library_setup=self.module_inputs, 
                filter_setup=self.module_filter, 
                verbose=self.verbose, 
                get_loss=False, 
                optimizer_threshold=self.threshold, 
                optimizer_alpha=self.regularization,
                )
            
            # test sindy agent and see if it breaks
            features_id = {}
            try:
                # set up SINDy-Agent
                agent_rnn.new_sess(experiment_list[id].session[0])
                self.sindy_agents[id] = AgentSindy(
                    sindy_models=sindy_models, 
                    n_actions=self.n_actions, 
                    beta_reward=agent_rnn._beta_reward,
                    beta_choice=agent_rnn._beta_choice, 
                    deterministic=True, 
                    counterfactual=agent_rnn._model._counterfactual,
                    )
                
                get_update_dynamics(experiment_list[id], self.sindy_agents[id])
                
                self.sindy_agents[id].new_sess()
        
                # save a dictionary of trained features per model
                features_id = {
                    'beta_reward': (('beta_reward'), (self.sindy_agents[id]._beta_reward)),
                    'beta_choice': (('beta_choice'), (self.sindy_agents[id]._beta_choice)),
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
                
                self.sindy_features[id] = deepcopy(features_id)
            except Exception as e:
                for m in sindy_models:
                    features_id[m] = (str(e), str(e))
                self.sindy_features[id] = deepcopy(features_id)
    
    def predict(self, conditions: np.ndarray):
        conditions_rnn = torch.tensor(conditions, dtype=torch.float32, device=self.device)        
        prediction_rnn = self.model_rnn(conditions_rnn, batch_first=True)[0].detach().cpu().numpy()
        
        prediction_sindy = np.zeros_like(prediction_rnn)
        sessions = np.unique(conditions[:, :, -1])
        for id in sessions:
            for trial in range(conditions.shape[1]):
                id_sindy = int(id) if list(self.sindy_agents.keys()) != [0] and np.sum(sessions) != 0 else 0
                prediction_sindy[int(id), trial] = self.sindy_agents[id_sindy].get_choice_probs()
                choice = np.argmax(conditions[int(id), trial, :self.n_actions])
                reward = conditions[int(id), trial, self.n_actions:2*self.n_actions]
                self.sindy_agents[id_sindy].update(choice, reward)
        return prediction_rnn, prediction_sindy