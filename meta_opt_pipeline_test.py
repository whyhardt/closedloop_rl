import sys
import os
import matplotlib.pyplot as plt
import warnings
import matplotlib.ticker as ticker
from numpy import arange
from torch import manual_seed
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

# Disable cudnn
from torch.backends import cudnn
cudnn.enabled = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn_t1t2
import pipeline_rnn_imaml
import pipeline_rnn_bo
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_dezfouli2019_blocks, RLRNN_eckstein2022#, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_FC


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------
path_model = 'params/T1-T2-R2-highLR.pkl'
path_data = 'data/eckstein2022/eckstein2022.csv'
train_test_ratio = 0.7
class_rnn = RLRNN_eckstein2022
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
# Meta-Optimization Test
# -------------------------------------------------------------------------------

# Set seed
manual_seed(6)

_, _, histories = pipeline_rnn_bo.main(
    
    dropout=0.25,
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

train_loss_history, val_loss_history, log_lambda_history = histories # Add hypergrad_history if not using Bayesian Optimization

epochs = arange(1, len(train_loss_history) + 1)

plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.subplot(1,3,2)
plt.plot(epochs, log_lambda_history, label='log_lambda')
# plt.yscale("log")
plt.xlabel('Epoch')
plt.ylabel('log_lambda')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# plt.subplot(1,3,3)
# plt.plot(epochs, hypergrad_history, label='Hypergrad')
# plt.xlabel('Epoch')
# plt.ylabel('Hypergrad')
# plt.legend()
# plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()
