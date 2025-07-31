import sys
import os
import matplotlib.pyplot as plt
import warnings
import matplotlib.ticker as ticker
import numpy as np
from torch import manual_seed
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")

# Disable cudnn
from torch.backends import cudnn
cudnn.enabled = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn_t1t2
import pipeline_rnn_imaml
import pipeline_rnn_awd
from resources.rnn_training_imaml import apply_mask
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_dezfouli2019_blocks, RLRNN_eckstein2022#, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_FC


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------
path_model = 'params/eckstein2022/iMAML_8192_2_eckstein2022_rnn.pkl'
path_data = 'data/eckstein2022/eckstein2022.csv'
train_test_ratio = 0.8
class_rnn = RLRNN_eckstein2022
additional_inputs = None

# -------------------------------------------------------------------------------
# Meta-Optimization Test
# -------------------------------------------------------------------------------

# Set seed for torch and np
seed = 10
manual_seed(seed)
np.random.seed(seed)

# model, _, histories = pipeline_rnn_awd.main(
model, _, histories = pipeline_rnn_imaml.main(
    
    dropout=0.25,
    train_test_ratio=train_test_ratio,
    
    # general training parameters
    checkpoint=False,
    epochs=8192, # <- 2^16
    scheduler=True,
    learning_rate=1e-2,

    # Meta-optimization parameters
    meta_update_interval=20,
    meta_lr=0.1,
    cg_steps=5,
    cg_damping=1e-3,
    init_lambda=5.0,

    # lambda_awd=0.1,  # Default from paper experiments

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

import numpy as np

train_loss_history, val_loss_history, log_lambda_history, hypergrad_history = histories
# train_loss_history, val_loss_history, log_lambda_history = histories

epochs = np.arange(1, len(train_loss_history) + 1)

# plt.figure(figsize=(18,5))
plt.figure(figsize=(12,5))

# Subplot 1: Losses
# plt.subplot(1,2,1)
plt.subplot(1,3,1)
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subplot 2: log_lambda
# plt.subplot(1,2,2)
plt.subplot(1,3,2)
plt.plot(epochs, log_lambda_history, label='λ_wd')
plt.xlabel('Epoch')
plt.ylabel('λ_wd')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subplot 3: hypergrad
plt.subplot(1,3,3)
plt.plot(epochs, hypergrad_history, label='Average Meta-Gradient Norm')
plt.xlabel('Epoch')
plt.ylabel('Average Meta-Gradient Norm')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()
