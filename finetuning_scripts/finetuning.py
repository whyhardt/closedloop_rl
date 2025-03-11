import time
import numpy as np
import pandas as pd
import torch
import random
import sys
import os
import optuna
import logging
import glob
import re
import json
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from resources.rnn_training import fit_model
from resources.rnn import RLRNN  
from resources.rnn_utils import DatasetRNN

###############################################################################
# Helper Functions

def compute_bic(loss, n_data, n_params):
    """
    Compute the Bayesian Information Criterion (BIC)
    """
    RSS = loss * n_data
    bic = n_data * np.log(RSS / n_data + 1e-8) + n_params * np.log(n_data)
    return bic

def compute_loss(model, dataset, n_steps=16, l1_weight_decay=1e-4):
    """
    Computes the average cross-entropy loss over the dataset.
    """
    model.eval()
    batch_size = min(32, len(dataset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0
    iterations = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    logging.info(f"Computing loss on dataset with {len(dataset)} sessions, batch size {batch_size}")

    with torch.no_grad():
        for xs, ys in dataloader:
            xs = xs.to(model.device)
            ys = ys.to(model.device)
            model.set_initial_state(batch_size=len(xs))
            state = model.get_state(detach=True)
            
            steps_loss = 0
            steps_count = 0
            for t in range(0, xs.shape[1], n_steps):
                n_block = min(xs.shape[1] - t, n_steps)
                xs_step = xs[:, t:t+n_block]
                ys_step = ys[:, t:t+n_block]
                
                logits, state = model(xs_step, state, batch_first=True)
                logits_2d = logits.reshape(-1, model._n_actions)
                target_1d = torch.argmax(ys_step.reshape(-1, model._n_actions), dim=1)
                loss_step = loss_fn(logits_2d, target_1d)
                steps_loss += loss_step.item()
                steps_count += 1
            
            total_loss += steps_loss / steps_count
            iterations += 1

    model.train()
    return total_loss / iterations

###############################################################################
# data load

def load_dataset_from_csv(file_path, dataset_index):
    """
    Load and prepare data from CSV file for RNN training.
    Each row in the CSV represents one trial, with session ID indicating the participant.
    
    Args:
        file_path: Path to the CSV file
        dataset_index: Which dataset to use
        
    Returns:
        A DatasetRNN object with prepared data tensors
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
    logging.info(f"Loading dataset from {file_path}")
    
    df = pd.read_csv(file_path)
    
    n_actions = 2
    
    n_participants = len(df['session'].unique())
    logging.info(f"Dataset has {n_participants} unique participants (sessions)")
    
    trials_per_participant = df.groupby('session').size().mean()
    logging.info(f"Average trials per participant: {trials_per_participant:.2f}")
    
    session_ids = df['session'].unique()
    
    logging.info(f"Dataset {dataset_index}: Processing {len(session_ids)} sessions")
    
    all_xs = []
    all_ys = []
    
    for session_id in session_ids:
        session_data = df[df['session'] == session_id]
        
        if len(session_data) == 0:
            continue
            
        # Extract features for inputs (xs)
        # Features to use based on the signals inr RLRNN
        xs_features = []
        
        # Create one-hot encoded choice/action
        choice = session_data['choice'].values
        choice_one_hot = np.zeros((len(choice), n_actions))
        for i, c in enumerate(choice):
            choice_one_hot[i, int(c)] = 1
            
        # Create one-hot encoded reward
        reward = session_data['reward'].values
        
        # Get learning parameters for this participant
        alpha_reward = session_data['alpha_reward'].values[0]
        alpha_penalty = session_data['alpha_penalty'].values[0]
        forget_rate = session_data['forget_rate'].values[0]
        beta_choice = session_data['beta_choice'].values[0]
        
        # For each trial, create a feature vector
        x_trials = []
        y_trials = []
        
        for idx in range(len(session_data)):
            # Get action values
            action_val_0 = session_data['action_value_0'].values[idx]
            action_val_1 = session_data['action_value_1'].values[idx]
            
            # Get reward values
            reward_val_0 = session_data['reward_value_0'].values[idx]
            reward_val_1 = session_data['reward_value_1'].values[idx]
            
            # Get choice values
            choice_val_0 = session_data['choice_value_0'].values[idx]
            choice_val_1 = session_data['choice_value_1'].values[idx]
            
            # Create trial feature vector based on the required signals
            # x_learning_rate_reward: alpha_reward
            # x_value_reward_not_chosen: reward value of non-chosen action
            # x_value_choice_chosen: choice value of chosen action
            # x_value_choice_not_chosen: choice value of non-chosen action
            # c_action: one-hot encoded choice
            # c_reward: reward value
            # c_value_reward: chosen action's reward value
            
            trial_choice = int(choice[idx])
            trial_features = np.array([
                alpha_reward,  # x_learning_rate_reward
                reward_val_1 if trial_choice == 0 else reward_val_0,  # x_value_reward_not_chosen
                choice_val_0 if trial_choice == 0 else choice_val_1,  # x_value_choice_chosen
                choice_val_1 if trial_choice == 0 else choice_val_0,  # x_value_choice_not_chosen
                choice_one_hot[idx, 0],  # c_action (first dimension)
                choice_one_hot[idx, 1],  # c_action (second dimension)
                reward[idx],  # c_reward
                reward_val_0 if trial_choice == 0 else reward_val_1,  # c_value_reward
                session_id  # Participant ID
            ])
            
            # Prepare next trial's action as target (one-hot encoded)
            if idx < len(session_data) - 1:
                next_choice = int(choice[idx + 1])
                target = np.zeros(n_actions)
                target[next_choice] = 1
            else:
                # For the last trial, just use the same choice (doesn't matter as it won't be used)
                target = np.zeros(n_actions)
                target[trial_choice] = 1
                
            x_trials.append(trial_features)
            y_trials.append(target)
            
        # Convert to tensors and add to dataset
        x_tensor = torch.tensor(x_trials, dtype=torch.float32).unsqueeze(0)  #  batch dimension
        y_tensor = torch.tensor(y_trials, dtype=torch.float32).unsqueeze(0)  # batch dimension
        
        all_xs.append(x_tensor)
        all_ys.append(y_tensor)
    
    # Combine all participant data
    dataset = DatasetRNN(
        torch.cat(all_xs, dim=0),
        torch.cat(all_ys, dim=0)
    )
    
    logging.info(f"Created dataset with shape: {dataset.xs.shape}")
    return dataset, n_participants

###############################################################################
# Cross-validation function

def run_fold_tuning(val_fold, data_files, dataset_indices, n_trials=3
, output_dir=None):
    """
    Run hyperparameter tuning for a single fold.
    
    Args:
        val_fold: The dataset index to use for validation
        data_files: List of data file paths
        dataset_indices: List of corresponding dataset indices
        n_trials: Number of trials for Optuna optimization
        output_dir: Directory to save results
        
    Returns:
        dict with best hyperparameters and validation loss
    """
    # Fixed hyperparameters
    hidden_size = 8
    batch_size = 32
    n_actions = 2
    
    train_files = []
    val_file = None
    
    for i, file_path in enumerate(data_files):
        file_index = dataset_indices[i]
        if file_index == val_fold:
            val_file = file_path
        else:
            train_files.append((file_path, file_index))
    
    logging.info(f"Validation file: {os.path.basename(val_file)}")
    logging.info(f"Training files: {[os.path.basename(f) for f, _ in train_files]}")
    
    val_dataset, n_participants_val = load_dataset_from_csv(val_file, val_fold)
    
    train_datasets = []
    max_participants = 0
    
    for train_file, file_index in train_files:
        train_dataset, n_participants = load_dataset_from_csv(train_file, file_index)
        train_datasets.append(train_dataset)
        max_participants = max(max_participants, n_participants)
    
    # Combine all training datasets
    if len(train_datasets) > 1:
        combined_xs = torch.cat([ds.xs for ds in train_datasets], dim=0)
        combined_ys = torch.cat([ds.ys for ds in train_datasets], dim=0)
        train_dataset = DatasetRNN(combined_xs, combined_ys)
        logging.info(f"Combined training dataset: {len(train_dataset)} sessions")
    else:
        train_dataset = train_datasets[0]
        logging.info(f"Using single training dataset: {len(train_dataset)} sessions")
    
    n_participants = max(max_participants, n_participants_val)
    logging.info(f"Using {n_participants} as maximum number of participants")
    
    n_trials_per_session = train_dataset.xs.shape[1]  
    n_data_val = val_dataset.xs.shape[0] * n_trials_per_session
    
    logging.info(f"Each session has {n_trials_per_session} trials")
    
    def objective(trial):
        trial_id = f"Fold {val_fold} - Trial {trial.number}"
        logging.info(f"Starting {trial_id}")
        
        dropout = trial.suggest_float('dropout', 0.1, 0.25)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        epochs_value = trial.suggest_categorical('epochs', [8])  # change to1024 
        l1_wd = trial.suggest_float('l1_weight_decay', 1e-6, 1e-3, log=True)
        convergence_threshold = trial.suggest_float('convergence_threshold', 1e-19, 1e-15, log=True)
        
        combo_str = (f"dropout={dropout:.3f}, lr={lr:.5f}, hidden_size={hidden_size}, "
                     f"epochs={epochs_value}, batch_size={batch_size}, l1_wd={l1_wd:.1e}, "
                     f"convergence_threshold={convergence_threshold:.1e}")
        logging.info(f"{trial_id} hyperparameters: {combo_str}")
        
        model_rnn = RLRNN(
            n_actions=n_actions,
            n_participants=n_participants,
            hidden_size=hidden_size,
            dropout=dropout,
            list_signals=[
                'x_learning_rate_reward',
                'x_value_reward_not_chosen',
                'x_value_choice_chosen',
                'x_value_choice_not_chosen',
                'c_action',
                'c_reward',
                'c_value_reward'
            ],
            device=torch.device('cpu')
        )
        optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=lr)
        
        t0 = time.time()
        model_rnn, optimizer_rnn, final_train_loss = fit_model(
            model=model_rnn,
            dataset_train=train_dataset,
            dataset_test=val_dataset,
            optimizer=optimizer_rnn,
            epochs=epochs_value,
            batch_size=batch_size,
            n_steps=16,
            l1_weight_decay=l1_wd,
            convergence_threshold=convergence_threshold,
            verbose=True,  
            scheduler=True
        )
        train_time = time.time() - t0
        final_lr = optimizer_rnn.param_groups[0]['lr']
        logging.info(f"{trial_id} completed training in {train_time:.2f}s, "
                     f"final train_loss: {final_train_loss:.7f}, final LR: {final_lr:.5f}")
        
        # Compute validation loss and BIC
        val_loss = compute_loss(model_rnn, val_dataset, n_steps=16, l1_weight_decay=l1_wd)
        n_params = sum(p.numel() for p in model_rnn.parameters())
        val_bic = compute_bic(val_loss, n_data_val, n_params)
        
        logging.info(f"{trial_id} validation loss: {val_loss:.7f}, BIC: {val_bic:.7f}")
        return val_loss
    
    logging.info(f"Starting Optuna optimization for fold {val_fold}...")
    logging.info("Explan: Using powers of 2 (2^n) for epochs to optimize LR scheduler behavior")
    logging.info("Explan: Using much lower convergence threshold (1e-19 to 1e-15) to prevent premature convergence")
    logging.info(f"Using fixed hyperparameters: hidden_size={hidden_size}, batch_size={batch_size}")
    
    # Create a study for this fold
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)  
    
    logging.info("==========================================")
    logging.info(f"Fold {val_fold} - Best Hyperparameters")
    logging.info("==========================================")
    logging.info(f"Best hyperparameters: {study.best_params}")
    
    trials_df = study.trials_dataframe()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fold_name = f"fold_{val_fold}"
        csv_filename = os.path.join(output_dir, f"trials_results_{fold_name}.csv")
        trials_df.to_csv(csv_filename, index=False)
    
    # Train final model for this fold using best parameters
    final_model = RLRNN(
        n_actions=n_actions,
        n_participants=n_participants,
        hidden_size=hidden_size,
        dropout=study.best_params['dropout'],
        list_signals=[
            'x_learning_rate_reward',
            'x_value_reward_not_chosen',
            'x_value_choice_chosen',
            'x_value_choice_not_chosen',
            'c_action',
            'c_reward',
            'c_value_reward'
        ],
        device=torch.device('cpu')
    )
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=study.best_params['learning_rate'])
    
    # Combine training and validation data for final model
    all_datasets = train_datasets + [val_dataset]
    if len(all_datasets) > 1:
        all_xs = torch.cat([ds.xs for ds in all_datasets], dim=0)
        all_ys = torch.cat([ds.ys for ds in all_datasets], dim=0)
        full_dataset = DatasetRNN(all_xs, all_ys)
        logging.info(f"Full dataset for final model: {len(full_dataset)} sessions")
    else:
        full_dataset = all_datasets[0]
    
    n_params = sum(p.numel() for p in final_model.parameters())
    logging.info(f"Final model has {n_params} trainable parameters")
    
    final_model, _, final_loss = fit_model(
        model=final_model,
        dataset_train=full_dataset,
        dataset_test=None,
        optimizer=final_optimizer,
        epochs=study.best_params['epochs'],
        batch_size=batch_size,
        n_steps=16,
        l1_weight_decay=study.best_params['l1_weight_decay'],
        convergence_threshold=study.best_params['convergence_threshold'],
        verbose=True, 
        scheduler=True
    )
    
    logging.info(f"Final model for fold {val_fold} completed with loss: {final_loss:.7f}")
    
    if output_dir:
        model_filename = os.path.join(output_dir, f"rnn_model_{fold_name}.pt")
        torch.save(final_model.state_dict(), model_filename)
        logging.info(f"Final model saved to {model_filename}")
    
    return {
        'fold': val_fold,
        'best_params': study.best_params,
        'best_val_loss': study.best_value,
        'n_params': n_params,
        'final_loss': final_loss
    }

###############################################################################
def main():
    # default to 3, change to 50+
    n_trials = 3
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            logging.warning(f"Invalid number of trials specified. Using {n_trials} instead.")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    all_data_files = glob.glob(os.path.join(data_dir, "data_rldm_*.csv"))
    
    if not all_data_files:
        raise ValueError(f"No data_rldm*.csv files found in {data_dir}")
    
    logging.info(f"Found {len(all_data_files)} dataset files: {[os.path.basename(f) for f in all_data_files]}")
    
    dataset_indices = []
    data_files = []
    for file_path in all_data_files:
        match = re.search(r'data_rldm_\d+p_(\d+)\.csv$', os.path.basename(file_path))
        if match:
            dataset_indices.append(int(match.group(1)))
            data_files.append(file_path)
    
    if not data_files:
        raise ValueError(f"No properly formatted data_rldm files found in {data_dir}")
    
    # Sort files by index for consistent behavior
    sorted_data = sorted(zip(dataset_indices, data_files))
    dataset_indices = [idx for idx, _ in sorted_data]
    data_files = [file_path for _, file_path in sorted_data]
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Starting {len(dataset_indices)}-fold cross-validation with {n_trials} trials per fold")
    
    all_results = []
    fold_params = {}
    
    for fold_idx in dataset_indices:
        logging.info(f"==== Starting fold {fold_idx} ====")
        result = run_fold_tuning(
            val_fold=fold_idx,
            data_files=data_files,
            dataset_indices=dataset_indices,
            n_trials=n_trials,
            output_dir=output_dir
        )
        all_results.append(result)
        fold_params[str(fold_idx)] = result['best_params']
    
    avg_params = defaultdict(float)
    param_counts = defaultdict(lambda: defaultdict(int))
    
    for result in all_results:
        for param, value in result['best_params'].items():
            avg_params[param] += value / len(all_results)
            
            # For categorical parameters, count occurrences
            if param == 'epochs':
                param_counts[param][value] += 1
    
    # For categorical params, use most common value instead of average
    for param, counts in param_counts.items():
        if counts:  # If we have counts for this parameter
            most_common_value = max(counts.items(), key=lambda x: x[1])[0]
            avg_params[param] = most_common_value
    
    logging.info("==========================================")
    logging.info("Cross-Validation Summary")
    logging.info("==========================================")
    
    # Print individual fold results
    for i, result in enumerate(all_results):
        logging.info(f"Fold {result['fold']} - Val Loss: {result['best_val_loss']:.7f}")
        
    # Calculate average validation loss
    avg_val_loss = sum(r['best_val_loss'] for r in all_results) / len(all_results)
    logging.info(f"Average validation loss across folds: {avg_val_loss:.7f}")
    
    logging.info("Average best hyperparameters across folds:")
    logging.info(avg_params)
    
    summary = {
        'fold_results': all_results,
        'average_val_loss': avg_val_loss,
        'average_params': dict(avg_params),
        'fold_params': fold_params
    }
    
    with open(os.path.join(output_dir, 'cross_validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logging.info(f"Cross-validation summary saved to {os.path.join(output_dir, 'cross_validation_summary.json')}")
    
    logging.info("==========================================")
    logging.info("Training Final Model with Average Parameters")
    logging.info("==========================================")
    logging.info(f"Parameters: {avg_params}")
    
    all_datasets = []
    max_participants = 0
    
    for file_path, file_index in zip(data_files, dataset_indices):
        dataset, n_participants = load_dataset_from_csv(file_path, file_index)
        all_datasets.append(dataset)
        max_participants = max(max_participants, n_participants)
    
    # Combine all datasets for final model
    all_xs = torch.cat([ds.xs for ds in all_datasets], dim=0)
    all_ys = torch.cat([ds.ys for ds in all_datasets], dim=0)
    full_dataset = DatasetRNN(all_xs, all_ys)
    logging.info(f"Final model dataset: {len(full_dataset)} sessions with {max_participants} max participants")
    
    # Create final model with averaged best parameters
    final_model = RLRNN(
        n_actions=2,
        n_participants=max_participants,
        hidden_size=8,  # Fixed
        dropout=avg_params['dropout'],
        list_signals=[
            'x_learning_rate_reward',
            'x_value_reward_not_chosen',
            'x_value_choice_chosen',
            'x_value_choice_not_chosen',
            'c_action',
            'c_reward',
            'c_value_reward'
        ],
        device=torch.device('cpu')
    )
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=avg_params['learning_rate'])
    
    n_params = sum(p.numel() for p in final_model.parameters())
    logging.info(f"Final model has {n_params} trainable parameters")
    
    final_model, _, final_loss = fit_model(
        model=final_model,
        dataset_train=full_dataset,
        dataset_test=None,
        optimizer=final_optimizer,
        epochs=int(avg_params['epochs']),
        batch_size=32,  # Fixed
        n_steps=16,
        l1_weight_decay=avg_params['l1_weight_decay'],
        convergence_threshold=avg_params['convergence_threshold'],
        verbose=True, 
        scheduler=True
    )
    
    logging.info(f"Final combined model training completed with loss: {final_loss:.7f}")
    
    model_filename = os.path.join(output_dir, "rnn_model_final.pt")
    torch.save(final_model.state_dict(), model_filename)
    logging.info(f"Final model saved to {model_filename}")

if __name__ == "__main__":
    main()