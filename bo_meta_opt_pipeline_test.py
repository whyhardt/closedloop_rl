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
import pipeline_rnn_bo
from resources.rnn_training_imaml import apply_mask
from resources.rnn import RLRNN, RLRNN_dezfouli2019, RLRNN_dezfouli2019_blocks, RLRNN_eckstein2022#, RLRNN_meta_eckstein2022, RLRNN_eckstein2022_FC


# -------------------------------------------------------------------------------
# SPICE CONFIGURATIONS
# -------------------------------------------------------------------------------
path_model = 'params/iMAML-4096.pkl'
path_data = 'data/eckstein2022/eckstein2022.csv'
train_test_ratio = 0.7
class_rnn = RLRNN
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

# Set seed for torch and np
seed = 10
manual_seed(seed)
np.random.seed(seed)

model, _, histories = pipeline_rnn_bo.main(
    
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

### Loss surface
import torch
import numpy as np
from torch.utils.data import DataLoader

def eval_val_loss_for_log_lambda(model, dataloader_val, log_lambda_value):
    model.eval()
    xs_val, ys_val = next(iter(dataloader_val))
    state = model.get_state(detach=True)
    preds = model(xs_val, state, batch_first=True)[0]
    preds = apply_mask(preds, xs_val)
    loss_fn = torch.nn.CrossEntropyLoss()
    l2_lambda = torch.exp(torch.tensor(log_lambda_value, device=xs_val.device))
    params = torch.cat([p.view(-1) for p in model.parameters()])
    l2_norm = (params ** 2).mean()
    val_loss = loss_fn(
        preds.reshape(-1, model._n_actions),
        torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1),
    ) + l2_lambda * l2_norm
    return val_loss.item()

dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=False)

log_lambda_range = np.linspace(-7, -1, 100)
val_losses = []
print("Calculating loss surface...")
for ll in log_lambda_range:
    val_loss = eval_val_loss_for_log_lambda(model, dataloader_val, ll)
    val_losses.append(val_loss)



train_loss_history, val_loss_history, log_lambda_history = histories

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

# Subplot 3: Loss landscape
plt.subplot(1,3,3)
plt.plot(log_lambda_range, val_losses)
plt.xlabel('log_lambda')
plt.ylabel('Validation Loss')
plt.title('Val Loss vs log_lambda')
plt.grid(True)

plt.tight_layout()
plt.show()
