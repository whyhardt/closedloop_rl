from typing import List, Union, Dict, Tuple
import numpy as np

import pysindy as ps

from resources.sindy_utils import remove_control_features, conditional_filtering
from resources.bandits import AgentNetwork, AgentSindy, BanditSession


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
            # optimizer=ps.STLSQ(threshold=optimizer_threshold, alpha=optimizer_alpha, verbose=True),
            optimizer=ps.SR3(thresholder="L0", threshold=optimizer_threshold, verbose=True),
            # optimizer=ps.ConstrainedSR3(thresholder="L1", threshold=optimizer_threshold, verbose=True),
            # optimizer=ps.SSR(criteria="model_residual"),
            feature_library=ps.PolynomialLibrary(polynomial_degree),
            discrete_time=True,
            feature_names=feature_names_i,
        )

        # fit sindy model
        sindy_models[x_feature].fit(x_train_i, u=control_i, t=1, multiple_trajectories=True, ensemble=False, library_ensemble=False)
        # post-process sindy weights
        for i in range(len(sindy_models[x_feature].model.steps[-1][1].coef_[0])):
            # case: coefficient is x_feature[k] 
            # --> Target in the case of non-available dynamics: 
            # x_feature[k+1] = 1.0 x_feature[k] and not e.g. x_feature[k+1] = 1.03 x_feature[k]
            if i == 1 and np.abs(1-sindy_models[x_feature].model.steps[-1][1].coef_[0, 1]) < optimizer_threshold:
                sindy_models[x_feature].model.steps[-1][1].coef_[0, 1] = 1.
            # case: any other coefficient
            elif i != 1 and np.abs(sindy_models[x_feature].model.steps[-1][1].coef_[0, i]) < optimizer_threshold:
                sindy_models[x_feature].model.steps[-1][1].coef_[0, i] = 0.
        if get_loss:
            loss_model = 1-sindy_models[x_feature].score(x_train_i, u=control_i, t=1, multiple_trajectories=True)
            loss += loss_model
            if verbose:
                print(f'Score for {x_feature}: {loss_model}')
        if verbose:
            sindy_models[x_feature].print()
    
    if get_loss:
        return sindy_models, loss
    else:
        return sindy_models
        
def setup_sindy_agent(
    update_rule, 
    n_actions: int,
    rnn,
    ):
    agent_sindy = AgentSindy(rnn, n_actions, deterministic=True)
    agent_sindy.set_update_rule(update_rule)
    
    return agent_sindy