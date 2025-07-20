import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy
from resources import rnn, sindy_utils

# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------

# path_data='data/parameter_recovery/data_256p_0.csv'
# path_model='params/parameter_recovery/rnn_256p_0.pkl'
# class_rnn = RLRNN
# sindy_config = SindyConfig

# Set seed for torch and np
from torch import manual_seed
import numpy as np
seed = 10
manual_seed(seed)
np.random.seed(seed)

path_data = 'data/dezfouli2019/dezfouli2019.csv'
path_model = 'params/dezfouli2019/AWD_8192_dezfouli2019_opt_rnn.pkl'
sindy_config = sindy_utils.SindyConfig_dezfouli2019
class_rnn = rnn.RLRNN_dezfouli2019
additional_inputs = None

# -------------------------------------------------------------------------------
# SPICE PIPELINE
# -------------------------------------------------------------------------------

agent_spice, features, loss = pipeline_sindy.main(
    
    class_rnn=class_rnn,
    model = path_model,
    data = path_data,
    additional_inputs_data=additional_inputs,
    save = True,
    
    # general recovery parameters
    participant_id=None,
    filter_bad_participants=False,
    use_optuna=True, # Set to True later
    pruning=False,
    
    # sindy parameters
    # optimizer_type="SR3_weighted_l1",
    train_test_ratio=0.8,
    polynomial_degree=1,
    optimizer_alpha=0.1,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=1,
    n_trials_same_action_off_policy=5,
    optuna_threshold=0.001,
    optuna_n_trials=50,
    verbose=False,
    
    # generated training dataset parameters
    n_actions=2,
    sigma=0.2,
    beta_reward=1.,
    alpha=0.25,
    alpha_penalty=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=False,
    get_loss=False,
    
    **sindy_config,
)

print(loss)