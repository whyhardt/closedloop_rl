# to jest dobre!!!

import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy
from resources.model_evaluation import log_likelihood, bayesian_information_criterion
from resources.bandits import get_update_dynamics
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import DatasetRNN


data_path = 'data/eckstein2022/eckstein2022.csv'
original_df = pd.read_csv(data_path)
unique_sessions = original_df['session'].unique()
slcn_path = '/Users/martynaplomecka/closedloop_rl/data/eckstein2022/SLCNinfo_Share.csv'
slcn_df = pd.read_csv(slcn_path)
columns_to_keep = ['ID', 'age - years']
available_columns = [col for col in columns_to_keep if col in slcn_df.columns]
slcn_df = slcn_df[available_columns]

# mapping using ID as key
slcn_mapping = {}
for _, row in slcn_df.iterrows():
    slcn_mapping[row['ID']] = row.to_dict()

behavior_metrics = []

for pid in tqdm(unique_sessions):
    participant_df = original_df[original_df['session'] == pid]
    if participant_df.empty:
        continue
    
    choices = participant_df['choice'].values
    n_switches = np.sum(np.abs(np.diff(choices)))
    switch_rate = n_switches / (len(choices) - 1) if len(choices) > 1 else 0
    
    # stay after reward rate
    stay_after_reward_count = 0
    stay_after_reward_total = 0
    for i in range(len(choices) - 1):
        current_choice = choices[i]
        next_choice = choices[i+1]
        current_reward = participant_df['reward'].iloc[i]
        if current_reward > 0:
            stay_after_reward_total += 1
            if next_choice == current_choice:
                stay_after_reward_count += 1
    stay_after_reward_rate = stay_after_reward_count / stay_after_reward_total if stay_after_reward_total > 0 else 0
    
    perseveration = np.mean(choices[:-1] == choices[1:]) if len(choices) > 1 else 0
    avg_reward = participant_df['reward'].mean()
    
    participant_data = {
        'participant_id': pid,
        'switch_rate': switch_rate,
        'stay_after_reward': stay_after_reward_rate,
        'perseveration': perseveration,
        'avg_reward': avg_reward,
        'n_trials': len(participant_df)
    }
    
    #  SLCN info
    if pid in slcn_mapping:
        slcn_info = slcn_mapping[pid]
        for key, value in slcn_info.items():
            if key == 'age - years':
                participant_data[f'slcn_{key}'] = value
            elif key == 'age_category':
                participant_data['age_category'] = value
    
    behavior_metrics.append(participant_data)

behavior_df = pd.DataFrame(behavior_metrics)

# Run SINDY pipeline
agent_spice, features, loss, participant_ids = pipeline_sindy.main(
    model='params/eckstein2022/rnn_eckstein2022_l1_0_0001_l2_0_0001.pkl',
    data='data/eckstein2022/eckstein2022.csv',
    save=True,
    participant_id=None,
    filter_bad_participants=True,
    use_optuna=False,
    polynomial_degree=1,
    optimizer_alpha=0.1,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=1,
    verbose=True,
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
    analysis=False,
    get_loss=True,
)

list_rnn_modules = [
    'x_learning_rate_reward',
    'x_value_reward_not_chosen',
    'x_value_choice_chosen',
    'x_value_choice_not_chosen'
]

# Map indices to real participant IDs
index_to_pid = {}
for i, idx in enumerate(unique_sessions):
    if i in participant_ids:
        index_to_pid[i] = idx

print(f"Total unique sessions: {len(unique_sessions)}")
print(f"Valid participants after filtering: {len(participant_ids)}")
print(f"Valid index to pid mappings: {len(index_to_pid)}")

# Get unique feature names across all SINDY modules
all_feature_names = set()
for module in list_rnn_modules:
    for idx in agent_spice._model.submodules_sindy[module]:
        sindy_model = agent_spice._model.submodules_sindy[module][idx]
        for name in sindy_model.get_feature_names():
            all_feature_names.add(f"{module}_{name}")

# Extract embeddings
embedding_matrix = agent_spice._model.participant_embedding.weight.detach().cpu().numpy()
embedding_size = embedding_matrix.shape[1]

# Create SINDY parameters dataframe
sindy_params = []
for idx in tqdm(participant_ids):
    if idx not in index_to_pid:
        print(f"Warning: Index {idx} not found in mapping")
        continue
    pid = index_to_pid[idx]
    param_dict = {'participant_id': pid}
    param_dict.update({feature: 0.0 for feature in all_feature_names})
    param_dict['beta_reward'] = features['beta_reward'].get(idx, 0.0)
    param_dict['beta_choice'] = features['beta_choice'].get(idx, 0.0)
    for module in list_rnn_modules:
        if idx in agent_spice._model.submodules_sindy[module]:
            model = agent_spice._model.submodules_sindy[module][idx]
            coefs = model.model.steps[-1][1].coef_.flatten()
            for i, name in enumerate(model.get_feature_names()):
                param_dict[f"{module}_{name}"] = coefs[i]
            param_dict[f"params_{module}"] = np.sum(np.abs(coefs) > 1e-10)
    param_dict['total_params'] = sum(param_dict.get(f"params_{m}", 0) for m in list_rnn_modules)
    if idx < embedding_matrix.shape[0]:
        for j in range(embedding_size):
            param_dict[f'embedding_{j}'] = embedding_matrix[idx, j]
    sindy_params.append(param_dict)
sindy_df = pd.DataFrame(sindy_params)

print(f"Number of participants in SINDY df: {len(sindy_df)}")
print(f"Number of participants in behavior df: {len(behavior_df)}")

# Merge SINDY and behavior dataframes
final_df = pd.merge(sindy_df, behavior_df, on='participant_id', how='left')
print(f"Number of participants in final merged df: {len(final_df)}")

missing_behavior = set(sindy_df['participant_id']) - set(behavior_df['participant_id'])
if missing_behavior:
    print(f"Warning: {len(missing_behavior)} participants in SINDY have no behavioral data")

# Add filtered_out flag
total_pids = set(behavior_df['participant_id'])
filtered_pids = total_pids - set(index_to_pid.values())
behavior_df['filtered_out'] = behavior_df['participant_id'].apply(lambda x: x in filtered_pids)

# Load dataset for model evaluation metrics
dataset_test, _, _, _ = convert_dataset(data_path)

from utils.setup_agents import setup_agent_rnn
agent_rnn = setup_agent_rnn(
    path_model='params/eckstein2022/rnn_eckstein2022_l1_0_0001_l2_0_0001.pkl',
    list_sindy_signals=list_rnn_modules + ['c_action', 'c_reward', 'c_value_reward', 'c_value_choice'],
)

# Calculate model evaluation metrics per participant
metrics_data = []
for idx in tqdm(participant_ids):
    if idx not in index_to_pid:
        continue
    real_pid = index_to_pid[idx]
    mask = dataset_test.xs[:, 0, -1] == idx
    if not mask.any():
        continue
    participant_data = DatasetRNN(*dataset_test[mask])

    # Reset agents, check with Daniel
    agent_spice.new_sess(participant_id=idx)
    agent_rnn.new_sess(participant_id=idx)

    # Fixed unpacking of get_update_dynamics
    _, probs_spice, _ = get_update_dynamics(
        experiment=participant_data.xs, agent=agent_spice
    )
    _, probs_rnn, _ = get_update_dynamics(
        experiment=participant_data.xs, agent=agent_rnn
    )
    n_trials_test = len(probs_spice)

    if n_trials_test == 0:
        continue

    ll_spice = log_likelihood(
        data=participant_data.ys[0, :n_trials_test].cpu().numpy(),
        probs=probs_spice
    )
    ll_rnn = log_likelihood(
        data=participant_data.ys[0, :n_trials_test].cpu().numpy(),
        probs=probs_rnn
    )

    spice_per_trial_likelihood = np.exp(ll_spice/(n_trials_test * agent_rnn._n_actions))
    rnn_per_trial_likelihood = np.exp(ll_rnn/(n_trials_test * agent_rnn._n_actions))

    n_params_dict = agent_spice.count_parameters(
        mapping_modules_values={m: 'x_value_choice' if 'choice' in m else 'x_value_reward'
                                for m in agent_spice._model.submodules_sindy}
    )
    if idx not in n_params_dict:
        continue
    n_parameters_spice = n_params_dict[idx]

    bic_spice = bayesian_information_criterion(
        data=participant_data.ys[0, :n_trials_test].cpu().numpy(),
        probs=probs_spice,
        n_parameters=n_parameters_spice
    )
    aic_spice = 2 * n_parameters_spice - 2 * ll_spice

    metrics_data.append({
        'participant_id': real_pid,
        'nll_spice': -ll_spice,
        'nll_rnn': -ll_rnn,
        'trial_likelihood_spice': spice_per_trial_likelihood,
        'trial_likelihood_rnn': rnn_per_trial_likelihood,
        'bic_spice': bic_spice,
        'aic_spice': aic_spice,
        'n_parameters_spice': n_parameters_spice,
        'metric_n_trials': n_trials_test
    })

metrics_df = pd.DataFrame(metrics_data)
print(f"Number of participants with metrics: {len(metrics_df)}")

final_df = pd.merge(final_df, metrics_df, on='participant_id', how='left')
print(f"Number of participants in final merged df with metrics: {len(final_df)}")

# Filter out participants older than 45 and younger than 8
age_filter = (final_df['slcn_age - years'] >= 8) & (final_df['slcn_age - years'] <= 45)
final_df = final_df[age_filter]

print(f"Number of participants after age filtering (8-45 years): {len(final_df)}")



final_df.to_csv('AAAAsindy_analysis_with_metrics.csv', index=False)
behavior_df.to_csv('behavior_metrics_with_filter_status.csv', index=False)