from typing import List, Union, Dict, Tuple
import numpy as np

import pysindy as ps

from sindy_utils import remove_control_features, conditional_filtering, optimize_beta as optimize_beta_func
from bandits import AgentNetwork, AgentSindy, BanditSession


def fit_model(
    x_train: List[np.ndarray], 
    control: List[np.ndarray] = None, 
    feature_names: List[str] = None, 
    library: ps.feature_library.base.BaseFeatureLibrary = None, 
    library_setup: Dict[str, List[str]] = {},
    filter_setup: Dict[str, Tuple[str, float]] = {},
    verbose: bool = False,
    get_loss: bool = False,
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
            
    # setup library
    if library is None:
        library = ps.PolynomialLibrary(degree=2)
    
    # train one sindy model per x_train variable instead of one sindy model for all
    sindy_models = {feature: None for feature in x_features}
    loss = 0
    # for i in range(x_train[0].shape[-1]):
    for i, x_feature in enumerate(x_features):
        if verbose:
            print(f'\nSINDy model for {x_feature}:')
        x_train_i = [x_sample[:, i].reshape(-1, 1) for x_sample in x_train]
        feature_names_i = [x_feature] + c_features
        if x_feature in filter_setup:
            x_train_i, control_i, feature_names_i = conditional_filtering(x_train_i, control, feature_names_i, filter_setup[x_feature][0], filter_setup[x_feature][1])
        else:
            control_i = control
        control_i = remove_control_features(control_i, feature_names_i[1:], library_setup[x_feature])
        feature_names_i = [x_feature] + library_setup[x_feature]
        if control_i is None:
            control_i = [np.zeros_like(x_train_i[0]) for _ in range(len(x_train_i))]
            feature_names_i = feature_names_i + ['u']
        # feature_names_i = [feature_names[i]] + feature_names[x_train[0].shape[-1]:]
        sindy_models[x_feature] = ps.SINDy(
            optimizer=ps.STLSQ(threshold=0.05, verbose=False, alpha=1e-1),
            feature_library=library,
            discrete_time=True,
            feature_names=feature_names_i,
        )
        
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
    ):
    agent_sindy = AgentSindy(n_actions)
    agent_sindy.set_update_rule(update_rule)
    if optimize_beta:
        beta = optimize_beta_func(experiment, comparison_agent, agent_sindy, plot=False)
        agent_sindy._beta = beta
        # print(f'Optimized SINDy-agent beta: {beta}')
        
    return agent_sindy