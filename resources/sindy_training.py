from typing import List, Union, Dict, Tuple
import numpy as np
from math import comb
import torch
from copy import deepcopy
from tqdm import tqdm

import pysindy as ps

from resources.sindy_utils import remove_control_features, conditional_filtering, create_dataset, remove_bad_participants
from resources.rnn_utils import DatasetRNN
from resources.bandits import AgentNetwork, AgentSpice, Bandits, BanditsDrift, AgentQ, create_dataset as create_dataset_bandits, get_update_dynamics
from resources.model_evaluation import bayesian_information_criterion as loss_metric
from resources.optimizer_selection import optimize_for_participant

def fit_sindy(
    variables: List[np.ndarray], 
    control: List[np.ndarray] = None, 
    rnn_modules: List[str] = None,
    control_signals: List[str] = None, 
    polynomial_degree: int = 1, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_type: str = "SR3_L1",
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 1,
    verbose: bool = False,
    ):
    
    # get all x-features
    x_features = rnn_modules
    # get all control features
    c_features = control_signals
    
    # make sure that all x_features are in the library_setup
    for feature in x_features:
        if feature not in library_setup:
            library_setup[feature] = []
    
    # train one sindy model per variable
    sindy_models = {feature: None for feature in x_features}
    loss = 0
    for index_feature, x_feature in enumerate(x_features):
        if verbose:
            print(f'\nSINDy model for {x_feature}:')
        
        # sort signals into corresponding arrays    
        x_i = [x[:, index_feature].reshape(-1, 1) for x in variables] # get current x-feature as target variable
        control_i = control
        
        feature_names_i = [x_feature] + c_features
        
        # filter target variable and control features according to filter conditions
        if x_feature in filter_setup:
            if not isinstance(filter_setup[x_feature][0], list):
                # check that filter_setup[x_feature] is a list of filter-conditions 
                filter_setup[x_feature] = [filter_setup[x_feature]]
            for filter_condition in filter_setup[x_feature]:
                x_i, control_i, feature_names_i = conditional_filtering(
                    x_train=x_i, 
                    control=control_i, 
                    feature_names=feature_names_i, 
                    feature_filter=filter_condition[0], 
                    condition=filter_condition[1], 
                    remove_feature_filter=False
                )
        
        # remove unnecessary control features according to library setup
        control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
        feature_names_i = [x_feature] + library_setup[x_feature]
        
        # add a dummy control feature if no control features are remaining - otherwise sindy breaks
        if control_i is None or len(control_i) == 0:
            raise NotImplementedError('Having no control signal in a module is currently not implemented')
            control_i = None
            feature_names_i = feature_names_i + ['dummy']
        
        # Set up increasing thresholds with polynomial degree for SR3_weighted_l1
        if optimizer_type == "SR3_weighted_l1":
            n_polynomial_combinations = np.array([comb(len(feature_names_i) + d, d) for d in range(polynomial_degree+1)])
            thresholds = np.zeros((1, n_polynomial_combinations[-1]))
            index = 0
            for d in range(len(n_polynomial_combinations)):
                thresholds[0, index:n_polynomial_combinations[d]] = d * optimizer_threshold
                index = n_polynomial_combinations[d]

        # Create optimizer based on type
        if optimizer_type == "STLSQ":
            optimizer = ps.STLSQ(alpha=optimizer_alpha, threshold=optimizer_threshold)
        elif optimizer_type == "SR3_L1":
            optimizer = ps.SR3(
                thresholder="L1",
                nu=optimizer_alpha,
                threshold=optimizer_threshold,
                verbose=verbose,
                max_iter=100
            )
        else:  # "SR3_weighted_l1" (default)
            optimizer = ps.SR3(
                thresholder="weighted_l1",
                nu=optimizer_alpha,
                threshold=optimizer_threshold,
                thresholds=thresholds,
                verbose=verbose,
                max_iter=100
            )
            
        # Setup sindy model for current x-feature
        sindy_models[x_feature] = ps.SINDy(
            optimizer=optimizer,
            feature_library=ps.PolynomialLibrary(polynomial_degree),
            discrete_time=True,
            feature_names=feature_names_i,
        )

        # fit sindy model
        sindy_models[x_feature].fit(x_i, u=control_i, t=1, multiple_trajectories=True, ensemble = True)
        
        if verbose:
            sindy_models[x_feature].print()
    
    return sindy_models
    
    
def fit_spice(
    rnn_modules: List[np.ndarray],
    control_signals: List[np.ndarray], 
    agent: AgentNetwork,
    data: DatasetRNN = None,
    polynomial_degree: int = 2, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_type: str = "SR3_L1",
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 0.1,
    participant_id: int = None,
    shuffle: bool = False,
    dataprocessing: Dict[str, List] = None,
    n_trials_off_policy: int = 1000,
    n_sessions_off_policy: int = 1,
    deterministic: bool = True,
    get_loss: bool = False,
    verbose: bool = False,
    use_optuna: bool = False,
    filter_bad_participants: bool = False,
    ) -> Tuple[AgentSpice, float]:
    """Fit a SPICE agent by replacing RNN modules with SINDy equations.

    Args:
        rnn_modules (List[np.ndarray]): List of RNN module names to be replaced with SINDy
        control_signals (List[np.ndarray]): List of control signal names
        agent (AgentNetwork): The trained RNN agent
        data (DatasetRNN, optional): Dataset for training/evaluation. Defaults to None.
        polynomial_degree (int, optional): Polynomial degree for SINDy. Defaults to 2.
        library_setup (Dict[str, List[str]], optional): Dictionary mapping features to library components. Defaults to {}.
        filter_setup (Dict[str, Tuple[str, float]], optional): Dictionary mapping features to filter conditions. Defaults to {}.
        optimizer_type (str, optional): Type of optimizer to use. Defaults to "SR3_L1".
        optimizer_threshold (float, optional): Threshold for optimizer. Defaults to 0.05.
        optimizer_alpha (float, optional): Alpha parameter for optimizer. Defaults to 1.
        participant_id (int, optional): Specific participant ID to process. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        dataprocessing (Dict[str, List], optional): Data processing options. Defaults to None.
        n_trials_off_policy (int, optional): Number of off-policy trials. Defaults to 2048.
        n_sessions_off_policy (int, optional): Number of off-policy sessions. Defaults to 1.
        deterministic (bool, optional): Whether to use deterministic mode. Defaults to True.
        get_loss (bool, optional): Whether to compute loss. Defaults to False.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        use_optuna (bool, optional): Whether to use Optuna for optimizer selection. Defaults to False.
        filter_bad_participants (bool, optional): Whether to filter out badly fitted participants. Defaults to False.

    Returns:
        Tuple[AgentSpice, List[int], float]: The SPICE agent, list of well-fitted participant IDs, and loss
    """
    
    if participant_id is not None:
        participant_ids = [participant_id]
    elif data is not None and participant_id is None:
        participant_ids = data.xs[..., -1].unique().int().cpu().numpy()
    else:
        raise ValueError("Either data or participant_id are required.")
        
    if n_sessions_off_policy > 0:
        # set up environment to create an off-policy dataset (w.r.t to trained RNN) of arbitrary length
        # The trained RNN will then perform value updates to get off-policy data
        environment = BanditsDrift(sigma=0.2, n_actions=agent._n_actions)
        
        # create a dummy dataset where each choice is chosen for n times and then an action switch occures
        xs_fit = torch.zeros((n_sessions_off_policy, n_trials_off_policy, 2*agent._n_actions+1)) - 1
        n_trials_same_action = 5
        for session in range(n_sessions_off_policy):
            # initialize first action
            current_action = torch.zeros(agent._n_actions)
            current_action[0] = 1
            for trial in range(n_trials_off_policy):
                current_action_index = torch.argmax(current_action).int().item()
                reward = torch.tensor(environment.step(current_action_index))
                xs_fit[session, trial, :-1] = torch.concat((current_action, reward))
                # action switch - go to next possible action item and if final go to first one
                if trial >= n_trials_same_action and trial % n_trials_same_action == 0:
                    current_action[current_action_index] = 0
                    current_action[current_action_index+1 if current_action_index+1 < len(current_action) else 0] = 1
                    
        # setup of dataset
        ys_fit = xs_fit[:, 1:, :agent._n_actions]
        xs_fit = xs_fit[:, :-1]
        dataset_fit = DatasetRNN(xs_fit, ys_fit)
        
        # repeat the off-policy data for every participant and add the corresponding participant ID
        xs_fit = dataset_fit.xs.repeat(len(participant_ids), 1, 1)
        ys_fit = dataset_fit.ys.repeat(len(participant_ids), 1, 1)
        # set participant ids correctly
        for index_pid in range(0, len(participant_ids)):
            xs_fit[n_sessions_off_policy*index_pid:n_sessions_off_policy*(index_pid+1):, :, -1] = participant_ids[index_pid]
        dataset_fit = DatasetRNN(xs=xs_fit, ys=ys_fit)
        
    elif n_sessions_off_policy <= 0 and data is not None:
        dataset_fit = data
        if participant_id is not None:
            mask_participant_id = dataset_fit.xs[:, 0, -1] == participant_id
            dataset_fit = DatasetRNN(*dataset_fit[mask_participant_id])
    elif n_sessions_off_policy <= 0 and data is None:
        raise ValueError("One of the arguments data or n_sessions_off_policy (> 0) must be given. If n_sessions_off_policy > 0 the SINDy modules will be fitted on the off-policy data regardless of data. If n_sessions_off_policy = 0 then data will be used to fit the SINDy modules.")
    
    sindy_models = {rnn_module: {} for rnn_module in rnn_modules}
    for pid in tqdm(participant_ids):
        # extract all necessary data from the RNN (memory state) and align with the control inputs (action, reward)
        mask_participant_id = dataset_fit.xs[:, 0, -1] == pid
        rnn_variables, control_parameters, _, _ = create_dataset(
            agent=agent,
            data=DatasetRNN(*dataset_fit[mask_participant_id]),
            rnn_modules=rnn_modules,
            control_signals=control_signals,
            shuffle=shuffle,
            dataprocessing=dataprocessing,
        )

        # If using Optuna, find the best optimizer configuration for this participant
        if use_optuna:
            
            # Find optimal optimizer and parameters for this participant
            optimizer_config = optimize_for_participant(
                variables=rnn_variables,
                control=control_parameters,
                rnn_modules=rnn_modules,
                control_signals=control_signals,
                library_setup=library_setup,
                filter_setup=filter_setup,
                polynomial_degree=polynomial_degree,
                n_trials=50,  # Adjust as needed
                verbose=verbose
            )
            
            # Use the optimized parameters for this participant
            pid_optimizer_type = optimizer_config["optimizer_type"]
            pid_optimizer_alpha = optimizer_config["optimizer_alpha"]
            pid_optimizer_threshold = optimizer_config["optimizer_threshold"]
            
            if verbose:
                print(f"\nUsing optimized parameters for participant {pid}:")
                print(f"  Optimizer type: {pid_optimizer_type}")
                print(f"  Alpha: {pid_optimizer_alpha}")
                print(f"  Threshold: {pid_optimizer_threshold}")
        else:
            # Use the global parameters
            pid_optimizer_type = optimizer_type
            pid_optimizer_alpha = optimizer_alpha
            pid_optimizer_threshold = optimizer_threshold

        # fit one SINDy-model per RNN-module
        sindy_models_id = fit_sindy(
            variables=rnn_variables,
            control=control_parameters,
            rnn_modules=rnn_modules,
            control_signals=control_signals,
            polynomial_degree=polynomial_degree,
            library_setup=library_setup,
            filter_setup=filter_setup,
            optimizer_type=pid_optimizer_type,
            optimizer_alpha=pid_optimizer_alpha,
            optimizer_threshold=pid_optimizer_threshold,
            verbose=verbose,
        )
            
        for rnn_module in rnn_modules:
            sindy_models[rnn_module][pid] = sindy_models_id[rnn_module]

    # set up a SINDy-based agent by replacing the RNN-modules with the respective SINDy-model
    agent_spice = AgentSpice(model_rnn=deepcopy(agent._model), sindy_modules=sindy_models, n_actions=agent._n_actions, deterministic=deterministic)
    
    # Initialize filtered_ids with all participant_ids
    filtered_ids = np.array(participant_ids)
    
    # Filter badly fitted participants if requested
    if filter_bad_participants and data is not None:
        agent_spice, filtered_ids = remove_bad_participants(
            agent_spice=agent_spice,
            agent_rnn=agent,
            dataset=data,
            participant_ids=participant_ids,
            verbose=verbose,
        )

    # compute loss
    loss = None
    if get_loss and data is None:
        raise ValueError("When get_loss is True, data must be given to compute the loss. Off-policy data won't be considered to compute the loss.")
    elif get_loss and data is not None:
        loss = 0
        n_trials_total = 0
        mapping_modules_values = {module: 'x_value_choice' if 'choice' in module else 'x_value_reward' for module in agent_spice._model.submodules_sindy}
        n_parameters = agent_spice.count_parameters(mapping_modules_values=mapping_modules_values)
        
        # Use filtered_ids for loss calculation if filtering was applied
        ids_to_use = filtered_ids if filter_bad_participants else participant_ids
        
        for pid in ids_to_use:
            if pid not in agent_spice._model.submodules_sindy[list(agent_spice._model.submodules_sindy.keys())[0]]:
                continue
                
            mask_participant_id = data.xs[:, 0, -1] == pid
            if not mask_participant_id.any():
                continue
                
            participant_data = DatasetRNN(*data[mask_participant_id])
            xs, ys = participant_data.xs.cpu().numpy(), participant_data.ys.cpu().numpy()
            
            probs = get_update_dynamics(experiment=xs, agent=agent_spice)[1]
            loss += loss_metric(data=ys[0, :len(probs)], probs=probs, n_parameters=n_parameters[pid])
            n_trials_total += len(probs)
            
        if n_trials_total > 0:
            loss = loss / n_trials_total
        else:
            loss = float('inf')  # If no valid trials, set loss to infinity

    return agent_spice, filtered_ids, loss