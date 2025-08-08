import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy
from resources import rnn, sindy_utils

# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------
class_rnn = rnn.RLRNN_eckstein2022
sindy_config = sindy_utils.SindyConfig_eckstein2022
additional_inputs = None

path_data = 'data/eckstein2022/eckstein2022.csv'
path_model = 'params/eckstein2022/iMAML_8192_05_eckstein2022_rnn.pkl'
train_test_ratio = 0.8

# class_rnn = rnn.RLRNN_dezfouli2019
# sindy_config = sindy_utils.SindyConfig_dezfouli2019
# additional_inputs = None

# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# path_model = 'params/dezfouli2019/T1-T2_8192_dezfouli2019_rnn_ep4096.pkl'
# train_test_ratio = [3, 6, 9]

# Set seed for torch and np
from torch import manual_seed
import numpy as np
seed = 10
manual_seed(seed)
np.random.seed(seed)

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
    use_optuna=False, # Set to True later
    pruning=False,
    
    # sindy parameters
    train_test_ratio=train_test_ratio,
    polynomial_degree=3,
    optimizer_alpha=0.1,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=1,
    n_trials_same_action_off_policy=5,
    optuna_threshold=0.1,
    optuna_n_trials=50,
    optimizer_type='SR3_weighted_l1',
    # optimizer_type='SR3_L1',
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
    
    analysis=True,
    get_loss=False,
    
    **sindy_config,
)

print(loss)