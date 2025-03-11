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
    Compute the Bayesian Information Criterion (BIC).
    """
    RSS = loss * n_data
    bic = n_data * np.log(RSS / n_data + 1e-8) + n_params * np.log(n_data)
    return bic

def compute_loss(model, dataset, n_steps=16, l1_weight_decay=1e-4):
    """
    Computes the average cross-entropy loss over the dataset.
    """
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    batch_size = min(32, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0
    iterations = 0
    loss_fn = torch.nn.CrossEntropyLoss()

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
                xs_step = xs[:, t : t + n_block]
                ys_step = ys[:, t : t + n_block]

                logits, state = model(xs_step, state, batch_first=True)
                logits_2d = logits.reshape(-1, model._n_actions)
                target_1d = torch.argmax(ys_step.reshape(-1, model._n_actions), dim=1)
                loss_step = loss_fn(logits_2d, target_1d)
                steps_loss += loss_step.item()
                steps_count += 1

            total_loss += steps_loss / steps_count
            iterations += 1

    model.train()
    if iterations == 0:
        return 0.0
    return total_loss / iterations

def load_dataset_from_csv(file_path, dataset_index=None):
    """
    Load and prepare data from CSV file for RNN training 
    Each row corresponds to one trial, with 'session' indicating participant ID.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    logging.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    n_actions = 2
    session_ids = df['session'].unique()
    if dataset_index is not None:
        logging.info(f"Dataset {dataset_index}: Found {len(session_ids)} unique participants")
    else:
        logging.info(f"Found {len(session_ids)} unique participants")

    all_xs = []
    all_ys = []

    for session_id in session_ids:
        session_data = df[df['session'] == session_id]
        if len(session_data) == 0:
            continue

        choice = session_data['choice'].values
        reward = session_data['reward'].values

        # Create one-hot for choice
        choice_one_hot = np.zeros((len(choice), n_actions))
        for i, c in enumerate(choice):
            choice_one_hot[i, int(c)] = 1

        x_trials = []
        y_trials = []

        for idx in range(len(session_data)):
            alpha_reward = session_data['alpha_reward'].values[idx]
            reward_val_0 = session_data['reward_value_0'].values[idx]
            reward_val_1 = session_data['reward_value_1'].values[idx]
            choice_val_0 = session_data['choice_value_0'].values[idx]
            choice_val_1 = session_data['choice_value_1'].values[idx]
            trial_choice = int(choice[idx])

            # 9-element feature vector
            trial_features = np.array([
                alpha_reward,
                reward_val_1 if trial_choice == 0 else reward_val_0,
                choice_val_0 if trial_choice == 0 else choice_val_1,
                choice_val_1 if trial_choice == 0 else choice_val_0,
                choice_one_hot[idx, 0],
                choice_one_hot[idx, 1],
                reward[idx],
                reward_val_0 if trial_choice == 0 else reward_val_1,
                session_id
            ])

            # Next trial's choice as target
            if idx < len(session_data) - 1:
                next_choice = int(choice[idx + 1])
                target = np.zeros(n_actions)
                target[next_choice] = 1
            else:
                # last trial: dummy
                target = np.zeros(n_actions)
                target[trial_choice] = 1

            x_trials.append(trial_features)
            y_trials.append(target)

        x_tensor = torch.tensor(x_trials, dtype=torch.float32).unsqueeze(0)  # shape (1, T, 9)
        y_tensor = torch.tensor(y_trials, dtype=torch.float32).unsqueeze(0)  # shape (1, T, 2)
        all_xs.append(x_tensor)
        all_ys.append(y_tensor)

    if len(all_xs) == 0:
        logging.warning("No data found for this CSV, returning empty dataset.")
        return DatasetRNN(torch.empty(0), torch.empty(0)), 0

    dataset = DatasetRNN(
        torch.cat(all_xs, dim=0),
        torch.cat(all_ys, dim=0)
    )
    logging.info(f"Created dataset with shape: {dataset.xs.shape}")
    return dataset, len(session_ids)

def split_dataset_last_20_percent(dataset, fraction=0.2):
    """
    Splits each session's trials into train (first 80%) and test (last 20%).
    """
    xs_train, ys_train = [], []
    xs_test, ys_test = [], []

    for i in range(len(dataset)):
        x_i, y_i = dataset[i]  # shapes: (T, 9), (T, 2)
        T = x_i.shape[0]
        cutoff = int(np.floor((1 - fraction) * T))
        if cutoff < 1:
            continue

        x_train, x_test = x_i[:cutoff], x_i[cutoff:]
        y_train, y_test = y_i[:cutoff], y_i[cutoff:]

        xs_train.append(x_train.unsqueeze(0))
        ys_train.append(y_train.unsqueeze(0))
        xs_test.append(x_test.unsqueeze(0))
        ys_test.append(y_test.unsqueeze(0))

    if len(xs_train) == 0:
        logging.warning("No valid sessions to split. Returning None.")
        return None, None

    train_x = torch.cat(xs_train, dim=0)
    train_y = torch.cat(ys_train, dim=0)
    test_x  = torch.cat(xs_test,  dim=0)
    test_y  = torch.cat(ys_test,  dim=0)

    train_dataset = DatasetRNN(train_x, train_y)
    test_dataset  = DatasetRNN(test_x,  test_y)
    return train_dataset, test_dataset

def main():
    """
    For each dataset CSV, we:
      1) Load it
      2) Split 80/20 (by session)
      3) Run Optuna to tune hyperparams
      4) Train final model on train set with best hyperparams
      5) Evaluate on validation set
    """

    n_trials = 50
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            logging.warning(f"Invalid n_trials specified. Using default {n_trials}.")

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    all_data_files = glob.glob(os.path.join(data_dir, "data_rldm_*.csv"))
    all_data_files = sorted(all_data_files)

    if not all_data_files:
        raise ValueError(f"No data_rldm_*.csv files found in {data_dir}")

    logging.info(f"Found {len(all_data_files)} files: {[os.path.basename(f) for f in all_data_files]}")

    results = []

    for file_path in all_data_files:
        dataset_name = os.path.basename(file_path)
        logging.info("=" * 70)
        logging.info(f"DATASET-SPECIFIC TRAINING for {dataset_name}")

        dataset, n_participants = load_dataset_from_csv(file_path)
        if len(dataset) == 0:
            logging.warning("Empty dataset, skipping.")
            continue

        train_dataset, val_dataset = split_dataset_last_20_percent(dataset, fraction=0.2)
        if train_dataset is None or val_dataset is None:
            logging.warning("Unable to split dataset. Skipping.")
            continue

        n_data_val = val_dataset.xs.shape[0] * val_dataset.xs.shape[1]
        hidden_size = 8
        n_actions = 2
        batch_size = 32

       
        def objective(trial):
            dropout = trial.suggest_float('dropout', 0.1, 0.25)
            lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            l1_wd = trial.suggest_float('l1_weight_decay', 1e-6, 1e-3, log=True)

            epochs_value = 4096
            conv_thr = 1e-19

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

            model_rnn, optimizer_rnn, final_train_loss = fit_model(
                model=model_rnn,
                dataset_train=train_dataset,
                dataset_test=val_dataset,
                optimizer=optimizer_rnn,
                epochs=epochs_value,
                batch_size=batch_size,
                n_steps=16,
                l1_weight_decay=l1_wd,
                convergence_threshold=conv_thr,
                verbose=False,
                scheduler=True
            )

            val_loss = compute_loss(model_rnn, val_dataset, n_steps=16, l1_weight_decay=l1_wd)
            n_params = sum(p.numel() for p in model_rnn.parameters())
            val_bic = compute_bic(val_loss, n_data_val, n_params)

            return val_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_val_loss = study.best_value
        logging.info(f"BEST PARAMS FOR {dataset_name}: {best_params}")
        logging.info(f"BEST VAL LOSS FOR {dataset_name}: {best_val_loss:.7f}")

        # Train final model with best hyperparams
        final_model = RLRNN(
            n_actions=n_actions,
            n_participants=n_participants,
            hidden_size=hidden_size,
            dropout=best_params['dropout'],
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
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

        epochs_value = 4096
        conv_thr = 1e-19

        final_model, _, final_loss = fit_model(
            model=final_model,
            dataset_train=train_dataset,
            dataset_test=None,  
            optimizer=final_optimizer,
            epochs=epochs_value,
            batch_size=batch_size,
            n_steps=16,
            l1_weight_decay=best_params['l1_weight_decay'],
            convergence_threshold=conv_thr,
            verbose=False,
            scheduler=True
        )

        # Evaluate final model on validation
        val_loss = compute_loss(
            final_model,
            val_dataset,
            n_steps=16,
            l1_weight_decay=best_params['l1_weight_decay']
        )
        n_params = sum(p.numel() for p in final_model.parameters())
        bic_score = compute_bic(val_loss, n_data_val, n_params)

        logging.info(f"Final validation loss for {dataset_name}: {val_loss:.7f}")
        logging.info(f"Final BIC for {dataset_name}: {bic_score:.7f}")

        output_dir = os.path.join(current_dir, "results_dataset_specific")
        os.makedirs(output_dir, exist_ok=True)
        model_filename = os.path.join(output_dir, f"rnn_model_{dataset_name}.pt")
        torch.save(final_model.state_dict(), model_filename)
        logging.info(f"Saved final model to {model_filename}")

        results.append({
            "dataset": dataset_name,
            "best_params": best_params,
            "best_val_loss": best_val_loss,
            "final_val_loss": val_loss,
            "bic": bic_score
        })

    summary_path = os.path.join(current_dir, "results_dataset_specific", "summary_dataset_specific.json")
    with open(summary_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    logging.info(f"Dataset-specific results saved to {summary_path}")


if __name__ == "__main__":
    main()
