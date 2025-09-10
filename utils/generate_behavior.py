import sys, os

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.bandits import create_dataset, get_update_dynamics, BanditsDrift, BanditsFlip_eckstein2022, Bandits_Standard, Agent, AgentQ
from resources.rnn_utils import DatasetRNN
from utils.setup_agents import setup_agent_rnn, setup_agent_spice
from utils.convert_dataset import convert_dataset

# dataset specific SPICE configurations and models
from resources.rnn import RLRNN, RLRNN_eckstein2022, RLRNN_dezfouli2019
from resources.sindy_utils import SindyConfig, SindyConfig_eckstein2022, SindyConfig_dezfouli2019
from benchmarking import benchmarking_dezfouli2019, benchmarking_eckstein2022


# ----------------------- GENERAL CONFIGURATION ----------------------------
agent_type = 'q_agent'  # 'rnn', 'spice', 'benchmark', 'baseline', 'q_agent'
n_trials_per_session = 200


# ------------------- CONFIGURATION ECKSTEIN2022 --------------------
dataset = 'eckstein2022'
benchmark_model = 'ApAnBrBcfBch'
baseline_model = 'ApBr'
class_rnn = RLRNN_eckstein2022
sindy_config = SindyConfig_eckstein2022
bandits_environment = BanditsDrift#BanditsFlip_eckstein2022
bandits_kwargs_per_session = [
    {'sigma': 0.2},
    ]
n_sessions = 1
setup_agent_benchmark = benchmarking_eckstein2022.setup_agent_benchmark
rl_model = benchmarking_eckstein2022.rl_model
path_rnn = f'params/{dataset}/rnn_{dataset}_l2_0_0001.pkl'
path_spice = f'params/{dataset}/spice_{dataset}_l2_0_0001.pkl'
path_benchmark = f'params/{dataset}/mcmc_{dataset}_BENCHMARK.nc'

# ------------------------ CONFIGURATION DEZFOULI2019 -----------------------
# dataset = 'dezfouli2019'
# benchmark_model = 'PhiChiBetaKappaC'
# baseline_model = 'PhiBeta'
# class_rnn = RLRNN_eckstein2022
# sindy_config = SindyConfig_eckstein2022
# bandits_environment = Bandits_Standard
# n_sessions = 6
# bandits_kwargs_per_session = [
#     {'reward_prob_0': 0.25, 'reward_prob_1': 0.05},
#     {'reward_prob_0': 0.125, 'reward_prob_1': 0.05},
#     {'reward_prob_0': 0.08, 'reward_prob_1': 0.05},
#     {'reward_prob_0': 0.05, 'reward_prob_1': 0.25},
#     {'reward_prob_0': 0.05, 'reward_prob_1': 0.125},
#     {'reward_prob_0': 0.05, 'reward_prob_1': 0.08},
#     ]
# setup_agent_benchmark = benchmarking_dezfouli2019.setup_agent_gql
# Dezfouli2019GQL = benchmarking_dezfouli2019.Dezfouli2019GQL
# path_rnn = f'params/{dataset}/rnn_{dataset}_l2_0_001.pkl'
# path_spice = f'params/{dataset}/spice_{dataset}_l2_0_001.pkl'
# path_benchmark = f'params/{dataset}/gql_{dataset}_BENCHMARK.pkl'


# ------------------- PIPELINE ----------------------------

path_data = f'data/{dataset}/{dataset}.csv'
path_save = f'data/{dataset}/{dataset}_generated_behavior_{agent_type}.csv'
if agent_type in ['baseline', 'benchmark']:
    path_benchmark = path_benchmark.replace('BENCHMARK', agent_type)

# check if generated behavior file exists
data_files = os.listdir(os.path.join(*path_save.split(os.path.sep)[:-1]))
count_files_generated = 0
for f in data_files:
    if path_save.split(os.path.sep)[-1].split('.')[0] in f:
        count_files_generated += 1
if count_files_generated > 0:
    path_save = path_save.split('.')[0] + f'_{count_files_generated}.csv'

if agent_type == 'spice':
    setup_agent = setup_agent_spice
elif agent_type == 'rnn':
    setup_agent = setup_agent_rnn
elif agent_type in ['baseline', 'benchmark']:
    setup_agent = setup_agent_benchmark
elif agent_type == 'q_agent':
    pass
else:
    raise ValueError(f'agent_type ({agent_type}) is unknown. Choose between one of: [baseline, benchmark, rnn, spice, q_agent].')

if path_data and agent_type != 'q_agent':
    n_participants = len(convert_dataset(path_data)[0].xs[:, 0, -1].unique())
else:
    n_participants = 128
    
dataset_xs, dataset_ys = [], []
for session in range(n_sessions):
    environment = bandits_environment(
        **bandits_kwargs_per_session[session],
        )

    if agent_type in ['rnn', 'spice', 'baseline', 'benchmark']:
        agent = setup_agent(
            class_rnn=class_rnn,
            path_rnn=path_rnn,
            path_spice=path_spice,
            path_model=path_benchmark,
            deterministic=False,
            model_config=benchmark_model if agent_type == 'benchmark' else baseline_model,
            )
    elif agent_type == 'q_agent':
        agent = AgentQ(
            alpha_reward=0.25, 
            beta_reward=3,
            alpha_penalty=-1,
            alpha_counterfactual_reward=0.,
            alpha_counterfactual_penalty=0.,
            beta_choice=0.,
            )
    else:
        raise ValueError(f'agent_type ({agent_type}) is unknown. Choose between one of: [baseline, benchmark, rnn, spice, q_agent].')
        
    if isinstance(agent, tuple):
        # in case of setup_agent_benchmark -> output: agent, n_parameters
        agent = agent[0]

    dataset = create_dataset(
                agent=agent,
                environment=environment,
                n_trials=n_trials_per_session,
                n_sessions=n_participants,
                verbose=False,
                )[0]
    
    dataset_xs.append(dataset.xs)
    dataset_ys.append(dataset.ys)
    
dataset = DatasetRNN(torch.concat(dataset_xs), torch.concat(dataset_ys))

# dataset columns
# general dataset columns
session, choice, reward = [], [], []

print('Saving values...')
n_actions = agent[0]._n_actions if isinstance(agent, list) else agent._n_actions
for i in tqdm(range(len(dataset))):    
    # get update dynamics
    experiment = dataset.xs[i].cpu().numpy()
    # qs, choice_probs, _ = get_update_dynamics(experiment, agent)
    
    # append behavioral data
    session += list(experiment[:, -1])
    choice += list(np.argmax(experiment[:, :n_actions], axis=-1))
    reward += list(np.max(experiment[:, n_actions:n_actions*2], axis=-1))
    
columns = ['session', 'choice', 'reward']
data = np.stack((np.array(session), np.array(choice), np.array(reward)), axis=-1)
df = pd.DataFrame(data=data, columns=columns)

df.to_csv(path_save, index=False)

print(f'Data saved to {path_save}')