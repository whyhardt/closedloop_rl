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
import pipeline_rnn_imaml_sindy
from resources.rnn_training_imaml3 import apply_mask
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_dezfouli2019_blocks, RLRNN_eckstein2022#, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_FC
from resources import sindy_utils

# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------
path_model = 'params/dezfouli2019/iMAMLS_TEST_dezfouli2019_rnn.pkl'
path_data = 'data/dezfouli2019/dezfouli2019.csv'
train_test_ratio = [3, 6, 9]
class_rnn = RLRNN_dezfouli2019
additional_inputs = None

sindy_config = sindy_utils.SindyConfig_dezfouli2019

# -------------------------------------------------------------------------------
# Meta-Optimization Test
# -------------------------------------------------------------------------------

# Set seed for torch and np
seed = 10
manual_seed(seed)
np.random.seed(seed)

model, _, histories = pipeline_rnn_imaml_sindy.main(
    
    dropout=0.25,
    train_test_ratio=train_test_ratio,
    
    # general training parameters
    checkpoint=False,
    epochs=8192, # <- 2^16
    scheduler=True,
    learning_rate=1e-2, # 1e-2

    # Meta-optimization parameters
    meta_update_interval=10,
    inner_steps=3,
    outer_lr=1e-1,
    hypergradient_steps=3,
    initial_reg_param=1e-4,

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

    # Pass SINDy config, use base SINDy parameters, so do not pass them
    **sindy_config,
)

import numpy as np

train_loss_history, val_loss_history, hparam_history = histories
epochs = np.arange(1, len(train_loss_history) + 1)

plt.figure(figsize=(12, 4))

# Plot 1: Losses
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_history, label="Train Loss", linewidth=2)
plt.plot(epochs, val_loss_history, label="Validation Loss", linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Regularization parameter
plt.subplot(1, 2, 2)
plt.plot(epochs, hparam_history, label="λ (Regularization)", linewidth=2, color='red')
plt.xlabel('Epoch')
plt.ylabel('Regularization Parameter')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
