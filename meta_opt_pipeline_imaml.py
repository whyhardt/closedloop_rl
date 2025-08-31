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
import pipeline_rnn_imaml
from resources.rnn_training_imaml2 import apply_mask
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_dezfouli2019_blocks, RLRNN_eckstein2022#, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_FC


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------
path_model = 'params/eckstein2022/NewiMAML.pkl'
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

model, _, histories = pipeline_rnn_imaml.main(
    
    dropout=0.25,
    train_test_ratio=train_test_ratio,
    
    # general training parameters
    checkpoint=False,
    epochs=8192, # <- 2^16
    scheduler=True,
    learning_rate=1e-2,

    # Meta-optimization parameters
    meta_update_interval=5,
    inner_steps=5,
    outer_lr=1e-1,
    hypergradient_steps=5,

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

train_loss_history, val_loss_history, hparam_history = histories

epochs = np.arange(1, len(train_loss_history) + 1)
tick_step = max(1, len(epochs) // 10) 

plt.figure(figsize=(18,5))

# Subplot 1: Losses
plt.subplot(1,3,1)
plt.plot(epochs, train_loss_history, label="Train Loss")
plt.plot(epochs, val_loss_history, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.xticks(np.arange(0, len(epochs) + 1, tick_step))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subplot 2: Regularization Weights (λ summary)
hparam_means = [h.mean() for h in hparam_history]
hparam_mins = [h.min() for h in hparam_history]
hparam_maxs = [h.max() for h in hparam_history]

plt.subplot(1,3,2)
plt.plot(epochs, hparam_means, label="Mean λ")
plt.plot(epochs, hparam_mins, label="Min λ", linestyle="--")
plt.plot(epochs, hparam_maxs, label="Max λ", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Regularization Weights (λ)")
plt.legend()
plt.xticks(np.arange(0, len(epochs) + 1, tick_step))
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Subplot 3: Distribution of λ’s (last epoch)
plt.subplot(1,3,3)
plt.hist(hparam_history[-1], bins=50, alpha=0.7)
plt.xlabel("λ value")
plt.ylabel("Frequency")
plt.title("Distribution of λ’s at final epoch")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
