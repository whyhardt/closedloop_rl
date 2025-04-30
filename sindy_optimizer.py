import numpy as np
from math import comb
import optuna
import pysindy as ps
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm

from resources.sindy_utils import remove_control_features, conditional_filtering, create_dataset
from resources.rnn_utils import DatasetRNN
from resources.bandits import AgentNetwork, AgentSpice, Bandits, BanditsDrift, AgentQ, create_dataset as create_dataset_bandits, get_update_dynamics
from resources.model_evaluation import akaike_information_criterion as loss_metric


def optimize_sindy_setup(
    variables: List[np.ndarray],
    control: List[np.ndarray],
    feature_names: List[str],
    polynomial_degree: int = 1,
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    n_trials: int = 20,
    timeout: Optional[int] = None,
    verbose: bool = False,
):
    """
    Use Optuna to find the best optimizer and hyperparameters for SINDy.
    
    Args:
        variables: List of numpy arrays containing the state variables
        control: List of numpy arrays containing the control signals
        feature_names: List of feature names
        polynomial_degree: Polynomial degree for the feature library
        library_setup: Dictionary mapping features to library components
        filter_setup: Dictionary mapping features to filter conditions
        n_trials: Number of optimization trials to run
        timeout: Timeout in seconds for optimization (None for no timeout)
        verbose: Whether to print verbose output
        
    Returns:
        tuple: (best_optimizer_type, best_params, best_score)
    """
    # Get all x-features
    x_features = [feature for feature in feature_names if feature.startswith('x_')]
    
    # Define objective function for Optuna
    def objective(trial):
        # Sample optimizer type
        optimizer_type = trial.suggest_categorical("optimizer_type", ["STLSQ", "SR3_L1", "SR3_weighted_l1"])
        
        # Sample hyperparameters
        optimizer_alpha = trial.suggest_float("optimizer_alpha", 0.01, 1.0, log=True)
        optimizer_threshold = trial.suggest_float("optimizer_threshold", 0.01, 0.2, log=True)
        
        # Initialize loss
        total_loss = 0
        
        # Train one SINDy model per variable
        for i, x_feature in enumerate(x_features):
            # Create training data for this feature
            x_i = [x.reshape(-1, 1) for x in variables[:, :, i]]
            x_to_control = variables[:, :, i != np.arange(variables.shape[-1])]
            control_i = [c for c in np.concatenate([x_to_control, control], axis=-1)] if control is not None and len(control) > 0 else []
            feature_names_i = [x_feature] + np.array(x_features)[i != np.arange(variables.shape[-1])].tolist() + \
                              [feature for feature in feature_names if feature.startswith('c_')]
            
            # Apply filters if specified
            if x_feature in filter_setup:
                if not isinstance(filter_setup[x_feature][0], list):
                    filter_setup[x_feature] = [filter_setup[x_feature]]
                for filter_condition in filter_setup[x_feature]:
                    x_i, control_i, feature_names_i = conditional_filtering(
                        x_i, control_i, feature_names_i, 
                        filter_condition[0], filter_condition[1], filter_condition[2]
                    )
            
            # Remove unnecessary control features
            if x_feature in library_setup:
                control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
                feature_names_i = [x_feature] + library_setup[x_feature]
            
            # Add dummy control if needed
            if control_i is None or len(control_i) == 0:
                control_i = [np.zeros_like(x_i[0]) for _ in range(len(x_i))]
                feature_names_i = feature_names_i + ['dummy']
            
            # Calculate thresholds for SR3
            n_polynomial_combinations = np.array([comb(len(feature_names_i) + d, d) for d in range(polynomial_degree+1)])
            thresholds = np.zeros((1, n_polynomial_combinations[-1]))
            index = 0
            for d in range(len(n_polynomial_combinations)):
                thresholds[0, index:n_polynomial_combinations[d]] = d * optimizer_threshold
                index = n_polynomial_combinations[d]
            
            # Create optimizer based on sampled type
            if optimizer_type == "STLSQ":
                optimizer = ps.STLSQ(alpha=optimizer_alpha, threshold=optimizer_threshold)
            elif optimizer_type == "SR3_L1":
                optimizer = ps.SR3(
                    thresholder="L1", 
                    nu=optimizer_alpha, 
                    threshold=optimizer_threshold, 
                    max_iter=100
                )
            else:  # SR3_weighted_l1
                optimizer = ps.SR3(
                    thresholder="weighted_l1", 
                    nu=optimizer_alpha, 
                    threshold=optimizer_threshold, 
                    thresholds=thresholds, 
                    max_iter=100
                )
            
            # Setup and fit SINDy model
            model = ps.SINDy(
                optimizer=optimizer,
                feature_library=ps.PolynomialLibrary(polynomial_degree),
                discrete_time=True,
                feature_names=feature_names_i,
            )
            
            try:
                model.fit(x_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False)
                
                # Calculate prediction error (MSE)
                predictions = []
                for j in range(len(x_i)):
                    if control_i is not None and len(control_i) > 0:
                        pred = model.predict(x_i[j], u=control_i[j])
                    else:
                        pred = model.predict(x_i[j])
                    predictions.append(pred)
                
                x_next = np.concatenate([x[1:] for x in x_i])
                pred_concat = np.concatenate([p[:-1] for p in predictions])
                mse = np.mean((x_next - pred_concat) ** 2)
                
                # Add to total loss
                total_loss += mse
                
                # Consider model complexity penalty (AIC-like)
                n_nonzero_coeffs = np.count_nonzero(model.coefficients())
                complexity_penalty = n_nonzero_coeffs * 0.01
                total_loss += complexity_penalty
                
            except Exception as e:
                # If model fitting fails, return a high loss
                if verbose:
                    print(f"Model fitting failed for {x_feature}: {e}")
                return float('inf')
        
        return total_loss
    
    # Create Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Get best parameters
    best_params = study.best_params
    best_optimizer_type = best_params.pop("optimizer_type")
    best_score = study.best_value
    
    if verbose:
        print(f"Best optimizer: {best_optimizer_type}")
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score}")
    
    return best_optimizer_type, best_params, best_score


def fit_sindy_optimized(
    variables: List[np.ndarray], 
    control: List[np.ndarray] = None, 
    feature_names: List[str] = None,
    polynomial_degree: int = 1, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_type: str = "auto",
    optimizer_params: Dict = None,
    optuna_trials: int = 20,
    optuna_timeout: Optional[int] = None,
    verbose: bool = False,
):
    """
    Fit SINDy models with optimized optimizer selection.
    
    Args:
        variables: List of numpy arrays containing the state variables
        control: List of numpy arrays containing the control signals
        feature_names: List of feature names
        polynomial_degree: Polynomial degree for the feature library
        library_setup: Dictionary mapping features to library components
        filter_setup: Dictionary mapping features to filter conditions
        optimizer_type: Type of optimizer to use ("auto", "STLSQ", "SR3_L1", "SR3_weighted_l1")
        optimizer_params: Dictionary of optimizer parameters
        optuna_trials: Number of optimization trials to run if optimizer_type is "auto"
        optuna_timeout: Timeout in seconds for optimization if optimizer_type is "auto"
        verbose: Whether to print verbose output
        
    Returns:
        dict: Dictionary of SINDy models for each feature
    """
    if feature_names is None:
        if len(library_setup) > 0:
            raise ValueError('If library_setup is provided, feature_names must be provided as well.')
        if len(filter_setup) > 0:
            raise ValueError('If datafilter_setup is provided, feature_names must be provided as well.')
        feature_names = [f'x{i}' for i in range(variables[0].shape[-1])]
    
    # Get all x-features
    x_features = [feature for feature in feature_names if feature.startswith('x_')]
    # Get all control features
    c_features = [feature for feature in feature_names if feature.startswith('c_')]
    
    # Make sure that all x_features are in the library_setup
    for feature in x_features:
        if feature not in library_setup:
            library_setup[feature] = []
    
    # Automatically select best optimizer if requested
    if optimizer_type == "auto":
        optimizer_type, optimizer_params, _ = optimize_sindy_setup(
            variables=variables,
            control=control,
            feature_names=feature_names,
            polynomial_degree=polynomial_degree,
            library_setup=library_setup,
            filter_setup=filter_setup,
            n_trials=optuna_trials,
            timeout=optuna_timeout,
            verbose=verbose,
        )
    elif optimizer_params is None:
        # Default parameters if none provided
        optimizer_params = {
            "optimizer_alpha": 0.1,
            "optimizer_threshold": 0.05
        }
    
    # Train one sindy model per variable
    sindy_models = {feature: None for feature in x_features}
    for i, x_feature in enumerate(x_features):
        if verbose:
            print(f'\nSINDy model for {x_feature}:')
        
        # Sort signals into corresponding arrays    
        x_i = [x.reshape(-1, 1) for x in variables[:, :, i]]  # get current x-feature as target variable
        x_to_control = variables[:, :, i != np.arange(variables.shape[-1])]  # get all other x-features as control variables
        control_i = [c for c in np.concatenate([x_to_control, control], axis=-1)] if control is not None and len(control) > 0 else []  # concatenate control variables with control features
        feature_names_i = [x_feature] + np.array(x_features)[i != np.arange(variables.shape[-1])].tolist() + c_features
        
        # Filter target variable and control features according to filter conditions
        if x_feature in filter_setup:
            if not isinstance(filter_setup[x_feature][0], list):
                # check that filter_setup[x_feature] is a list of filter-conditions 
                filter_setup[x_feature] = [filter_setup[x_feature]]
            for filter_condition in filter_setup[x_feature]:
                x_i, control_i, feature_names_i = conditional_filtering(
                    x_i, control_i, feature_names_i, 
                    filter_condition[0], filter_condition[1], filter_condition[2]
                )
        
        # Remove unnecessary control features according to library setup
        control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
        feature_names_i = [x_feature] + library_setup[x_feature]
        
        # Add a dummy control feature if no control features are remaining
        if control_i is None or len(control_i) == 0:
            control_i = [np.zeros_like(x_i[0]) for _ in range(len(x_i))]
            feature_names_i = feature_names_i + ['dummy']
        
        # Set up increasing thresholds with polynomial degree
        n_polynomial_combinations = np.array([comb(len(feature_names_i) + d, d) for d in range(polynomial_degree+1)])
        thresholds = np.zeros((1, n_polynomial_combinations[-1]))
        index = 0
        for d in range(len(n_polynomial_combinations)):
            thresholds[0, index:n_polynomial_combinations[d]] = d * optimizer_params["optimizer_threshold"]
            index = n_polynomial_combinations[d]
        
        # Create optimizer based on type
        if optimizer_type == "STLSQ":
            optimizer = ps.STLSQ(
                alpha=optimizer_params["optimizer_alpha"], 
                threshold=optimizer_params["optimizer_threshold"]
            )
        elif optimizer_type == "SR3_L1":
            optimizer = ps.SR3(
                thresholder="L1", 
                nu=optimizer_params["optimizer_alpha"], 
                threshold=optimizer_params["optimizer_threshold"], 
                verbose=verbose, 
                max_iter=100
            )
        else:  # SR3_weighted_l1
            optimizer = ps.SR3(
                thresholder="weighted_l1", 
                nu=optimizer_params["optimizer_alpha"], 
                threshold=optimizer_params["optimizer_threshold"], 
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
        
        # Fit sindy model
        sindy_models[x_feature].fit(x_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False)
        
        if verbose:
            sindy_models[x_feature].print()
    
    return sindy_models


def fit_spice_optimized(
    rnn_modules: List[str],
    control_parameters: List[str], 
    agent: AgentNetwork,
    data: DatasetRNN = None,
    polynomial_degree: int = 2, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    optimizer_type: str = "auto",
    optimizer_params: Dict = None,
    optuna_trials: int = 20,
    optuna_timeout: Optional[int] = None,
    participant_id: int = None,
    shuffle: bool = False,
    dataprocessing: Dict[str, List] = None,
    n_trials_off_policy: int = 2048,
    n_sessions_off_policy: int = 1,
    deterministic: bool = True,
    get_loss: bool = False,
    verbose: bool = False,
) -> Tuple[AgentSpice, float]:
    """
    Fit a SPICE agent with optimized SINDy models.
    
    Args:
        rnn_modules: List of RNN module names
        control_parameters: List of control parameters
        agent: Agent network
        data: Dataset for RNN
        polynomial_degree: Polynomial degree for the feature library
        library_setup: Dictionary mapping features to library components
        filter_setup: Dictionary mapping features to filter conditions
        optimizer_type: Type of optimizer to use ("auto", "STLSQ", "SR3_L1", "SR3_weighted_l1")
        optimizer_params: Dictionary of optimizer parameters
        optuna_trials: Number of optimization trials to run if optimizer_type is "auto"
        optuna_timeout: Timeout in seconds for optimization if optimizer_type is "auto"
        participant_id: Participant ID
        shuffle: Whether to shuffle the data
        dataprocessing: Dictionary of data processing options
        n_trials_off_policy: Number of trials for off-policy data
        n_sessions_off_policy: Number of sessions for off-policy data
        deterministic: Whether to use deterministic mode
        get_loss: Whether to compute loss
        verbose: Whether to print verbose output
        
    Returns:
        tuple: (AgentSpice, loss)
    """
    if participant_id is not None:
        participant_ids = [participant_id]
    elif data is not None and participant_id is None:
        participant_ids = data.xs[..., -1].unique().int().cpu().numpy()
    else:
        raise ValueError("Either data or participant_id are required.")
        
    if n_sessions_off_policy > 0:
        # Set up environment to create an off-policy dataset
        environment = BanditsDrift(sigma=0.2, n_actions=agent._n_actions)
        agent_dummy = AgentQ(n_actions=agent._n_actions, alpha_reward=0.5, beta_reward=1.0)
        dataset_fit = create_dataset_bandits(agent=agent_dummy, environment=environment, n_trials=n_trials_off_policy, n_sessions=n_sessions_off_policy)[0]
        # Repeat the off-policy data for every participant and add the corresponding participant ID
        xs_fit = dataset_fit.xs.repeat(len(participant_ids), 1, 1)
        ys_fit = dataset_fit.ys.repeat(len(participant_ids), 1, 1)
        # Set participant ids correctly
        for index_pid in range(0, len(participant_ids)):
            xs_fit[n_sessions_off_policy*index_pid:n_sessions_off_policy*(index_pid+1):, :, -1] = participant_ids[index_pid]
        dataset_fit = DatasetRNN(xs=xs_fit, ys=ys_fit)
        
    elif n_sessions_off_policy <= 0 and data is not None:
        dataset_fit = data
        if participant_id is not None:
            mask_participant_id = dataset_fit.xs[:, 0, -1] == participant_id
            dataset_fit = DatasetRNN(*dataset_fit[mask_participant_id])
    elif n_sessions_off_policy == 0 and data is None:
        raise ValueError("One of the arguments data or n_sessions_off_policy (> 0) must be given.")
    
    sindy_models = {rnn_module: {} for rnn_module in rnn_modules}
    for pid in tqdm(participant_ids):
        # Extract all necessary data from the RNN (memory state) and align with the control inputs
        mask_participant_id = dataset_fit.xs[:, 0, -1] == pid
        variables, control_parameters_data, feature_names, _ = create_dataset(
            agent=agent,
            data=DatasetRNN(*dataset_fit[mask_participant_id]),
            participant_id=pid,
            shuffle=shuffle,
            dataprocessing=dataprocessing,
        )

        # Fit one SINDy-model per RNN-module with optimized optimizer selection
        sindy_models_id = fit_sindy_optimized(
            variables=variables,
            control=control_parameters_data,
            feature_names=feature_names,
            polynomial_degree=polynomial_degree,
            library_setup=library_setup,
            filter_setup=filter_setup,
            optimizer_type=optimizer_type,
            optimizer_params=optimizer_params,
            optuna_trials=optuna_trials,
            optuna_timeout=optuna_timeout,
            verbose=verbose,
        )
        
        for rnn_module in rnn_modules:
            sindy_models[rnn_module][pid] = sindy_models_id[rnn_module]

    # Set up a SINDy-based agent
    agent_spice = AgentSpice(model_rnn=deepcopy(agent._model), sindy_modules=sindy_models, n_actions=agent._n_actions, deterministic=deterministic)
    
    # Compute loss
    loss = None
    if get_loss and data is None:
        raise ValueError("When get_loss is True, data must be given to compute the loss.")
    elif get_loss and data is not None:
        loss = 0
        n_trials_total = 0
        mapping_modules_values = {module: 'x_value_choice' if 'choice' in module else 'x_value_reward' for module in agent_spice._model.submodules_sindy}
        n_parameters = agent_spice.count_parameters(mapping_modules_values=mapping_modules_values)
        for pid in participant_ids:
            xs, ys = data.xs.cpu().numpy(), data.ys.cpu().numpy()
            probs = get_update_dynamics(experiment=xs[pid], agent=agent_spice)[1]
            loss += loss_metric(data=ys[pid, :len(probs)], probs=probs, n_parameters=n_parameters[pid]) 
            n_trials_total += len(probs)
        loss = loss/n_trials_total
    
    return agent_spice, loss