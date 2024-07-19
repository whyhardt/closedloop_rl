from typing import List, Union, Dict, Tuple
import numpy as np

import pysindy as ps

from sindy_utils import remove_control_features, conditional_filtering, optimize_beta as optimize_beta_func
from bandits import AgentNetwork, AgentSindy, BanditSession


def fit_model(
    x_train: List[np.ndarray], 
    control: List[np.ndarray] = None, 
    feature_names: List[str] = None, 
    polynomial_degree: int = 1, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    verbose: bool = False,
    get_loss: bool = False,
    optimizer_threshold: float = 0.05,
    optimizer_alpha: float = 1e-1
    ):
    
    if feature_names is None:
        if len(library_setup) > 0:
            raise ValueError('If library_setup is provided, feature_names must be provided as well.')
        if len(filter_setup) > 0:
            raise ValueError('If datafilter_setup is provided, feature_names must be provided as well.')
        feature_names = [f'x{i}' for i in range(x_train[0].shape[-1])]
    # get all x-features
    x_features = [feature for feature in feature_names if feature.startswith('x')]
    # get all control features
    c_features = [feature for feature in feature_names if feature.startswith('c')]
    
    # make sure that all x_features are in the library_setup
    for feature in x_features:
        if feature not in library_setup:
            library_setup[feature] = []
            
    x_train = np.stack(x_train, axis=0)
    control = np.stack(control, axis=0)
    
    # train one sindy model per x_train variable instead of one sindy model for all
    sindy_models = {feature: None for feature in x_features}
    loss = 0
    for i, x_feature in enumerate(x_features):
        if verbose:
            print(f'\nSINDy model for {x_feature}:')
        
        # sort signals into corresponding arrays    
        x_train_i = [x.reshape(-1, 1) for x in x_train[:, :, i]]  # get current x-feature as target variable
        x_to_control = x_train[:, :, i != np.arange(x_train.shape[-1])]  # get all other x-features as control variables
        control_i = [c for c in np.concatenate([x_to_control, control], axis=-1)]  # concatenate control variables with control features
        feature_names_i = [x_feature] + np.array(x_features)[i != np.arange(x_train.shape[-1])].tolist() + c_features
        
        # filter target variable and control features according to filter conditions
        if x_feature in filter_setup:
            if not isinstance(filter_setup[x_feature][0], list):
                # check that filter_setup[x_feature] is a list of filter-conditions 
                filter_setup[x_feature] = [filter_setup[x_feature]]
            else:
                print('okay')
            for filter_condition in filter_setup[x_feature]:
                x_train_i, control_i, feature_names_i = conditional_filtering(x_train_i, control_i, feature_names_i, filter_condition[0], filter_condition[1], filter_condition[2])
        
        # remove unnecessary control features according to library setup
        control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
        feature_names_i = [x_feature] + library_setup[x_feature]
        
        # add a dummy control feature if no control features are remaining
        if control_i is None:
            control_i = [np.zeros_like(x_train_i[0]) for _ in range(len(x_train_i))]
            feature_names_i = feature_names_i + ['u']
        
        # setup sindy model for current x-feature
        sindy_models[x_feature] = ps.SINDy(
            # optimizer=ps.STLSQ(threshold=0.03, verbose=True, alpha=optimizer_alpha),
            optimizer=ps.SR3(threshold=0.03, thresholder="L0"),
            # optimizer=ps.SSR(),
            feature_library=ps.PolynomialLibrary(polynomial_degree),
            discrete_time=True,
            feature_names=feature_names_i,
        )
        
        # fit sindy model
        sindy_models[x_feature].fit(x_train_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False, library_ensemble=False)
        if get_loss:
            loss_model = 1-sindy_models[x_feature].score(x_train_i, u=control_i, t=1, multiple_trajectories=True)
            loss += loss_model
            if verbose:
                print(f'Loss for {x_feature}: {loss_model}')
        if verbose:
            sindy_models[x_feature].print()
    
    if get_loss:
        return sindy_models, loss
    else:
        return sindy_models
        
def setup_sindy_agent(
    update_rule, 
    n_actions: int = None,
    optimize_beta: bool = False,
    experiment: BanditSession = None,
    comparison_agent: AgentNetwork = None,
    verbose: bool = False
    ):
    agent_sindy = AgentSindy(n_actions, deterministic=True)
    agent_sindy.set_update_rule(update_rule)
    if optimize_beta:
        beta = optimize_beta_func(experiment, comparison_agent, agent_sindy, plot=False)
        agent_sindy._beta = beta
        if verbose:
            print(f'Optimized SINDy-agent beta: {beta}')
        
    return agent_sindy