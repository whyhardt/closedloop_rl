import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_dezfouli2019_blocks, RLRNN_eckstein2022#, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_FC


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------

path_model = 'params/T1-T2-TEST.pkl'
path_data = 'data/dezfouli2019/dezfouli2019.csv'
train_test_ratio = [3, 6, 9]  # list of test sessions
class_rnn = RLRNN_dezfouli2019
additional_inputs = None
# class_rnn = RLRNN_meta_eckstein2022
# additional_inputs = ['age']

# class_rnn = RLRNN_dezfouli2019
# train_test_ratio = None#[3, 6, 9]#[1, 3, 4, 6, 8, 10]   # list of test sessions
# path_model = 'params/dezfouli2019/rnn_dezfouli2019_no_l1_l2_0_crossval_baseline.pkl'
# path_data = 'data/dezfouli2019/dezfouli2019_simulated_gql_multi_session_d1.csv'
# additional_inputs = None

# class_rnn = RLRNN_dezfouli2019_blocks
# train_test_ratio = [1, 3, 4, 6, 8, 10]  # list of test sessions
# path_model = 'params/dezfouli2019/rnn_dezfouli2019_blocks_rldm_l1emb_0_001_l2_0_0001.pkl'
# path_data = 'data/dezfouli2019/dezfouli2019.csv'
# additional_inputs = None

# -------------------------------------------------------------------------------
# SPICE PIPELINE
# -------------------------------------------------------------------------------
from torch import manual_seed
import numpy as np

manual_seed(10)
np.random.seed(10)

decays_loss = []

for weight_decay in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:

    _, loss = pipeline_rnn.main(
        
        # sparsification parameter
        # generalization parameters
        l2_weight_decay=weight_decay,
        train_test_ratio=train_test_ratio,
        
        # general training parameters
        checkpoint=False,
        epochs=1024, # <- 2^16
        scheduler=True,
        learning_rate=1e-2,
        
        # hand-picked params
        n_steps=-1,
        embedding_size=32,
        batch_size=-1,
        sequence_length=-1,
        bagging=True,
        
        class_rnn=class_rnn,
        model=path_model,
        data=path_data,
        additional_inputs_data=additional_inputs,
        
        # synthetic dataset parameters
        n_sessions=128,
        n_trials=200,
        sigma=0.2,
        beta_reward=3.,
        alpha_reward=0.25,
        alpha_penalty=0.5,
        forget_rate=0.,
        confirmation_bias=0.,
        beta_choice=0.,
        alpha_choice=1.,
        counterfactual=False,
        alpha_counterfactual=0.,
        
        save_checkpoints=True,
        analysis=False,
        participant_id=0,
    )

    decays_loss.append(loss)

# Print the results
print("Weight Decay vs Loss:")
for wd, loss in zip([0.0, 1e-4, 1e-3, 1e-2, 1e-1], decays_loss):
    print(f"Weight Decay: {wd}, Loss: {loss:.4f}")
