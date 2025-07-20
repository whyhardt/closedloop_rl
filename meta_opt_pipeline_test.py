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
path_model = 'params/dezfouli2019/AWD_8192_dezfouli2019_rnn.pkl'
path_data = 'data/dezfouli2019/dezfouli2019.csv'
train_test_ratio = [3, 6, 9]
class_rnn = RLRNN_dezfouli2019
additional_inputs = None

# -------------------------------------------------------------------------------
# Meta-Optimization Test
# -------------------------------------------------------------------------------

# Set seed for torch and np
seed = 10
manual_seed(seed)
np.random.seed(seed)

model, _, histories = pipeline_rnn_awd.main( # Set iMAML or T1-T2 or AWD pipeline
    
    dropout=0.25,
    train_test_ratio=train_test_ratio,
    
    # general training parameters
    checkpoint="params/dezfouli2019/AWD_8192_dezfouli2019_rnn_ep4096.pkl",
    epochs=4096, # <- 2^16
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

### Loss surface
import torch
import numpy as np

train_loss_history, val_loss_history, log_lambda_history, hypergrad_history = histories

epochs = np.arange(1, len(train_loss_history) + 1)

plt.figure(figsize=(18,5))

# Subplot 1: Losses
plt.subplot(1,3,1)
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subplot 2: log_lambda
plt.subplot(1,3,2)
plt.plot(epochs, log_lambda_history, label='log_lambda')
plt.xlabel('Epoch')
plt.ylabel('log_lambda')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subplot 3: hypergrad
plt.subplot(1,3,3)
plt.plot(epochs, hypergrad_history, label='Hypergrad')
plt.xlabel('Epoch')
plt.ylabel('Hypergrad')
plt.legend()
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.show()
