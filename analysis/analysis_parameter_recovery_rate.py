import os, sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.setup_agents import setup_agent_sindy


# create a mapping of ground truth parameters to library parameters
mapping_x_V_LR = {
    '1': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (alpha_penalty+confirmation_bias*0.25)*beta_reward,
    
    'x_V_LR': 0,
    
    'c_r': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (-alpha_reward-0.5*confirmation_bias)*beta_reward,
    
    'c_V': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (-0.5*confirmation_bias),
    
    'x_V_LR c_V': 0,
    'c_V x_V_LR': 0,
    
    'x_V_LR c_r': 0,
    'c_r x_V_LR': 0,
    
    'x_V_LR^2': 0,
    
    'c_V c_r': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (confirmation_bias)*beta_reward,
    'c_r c_V': lambda alpha_reward, alpha_penalty, confirmation_bias, beta_reward: (confirmation_bias)*beta_reward,
    
    'c_V^2': 0,
    
    'c_r^2': 0,
}

mapping_x_V_nc = {
    '1': lambda forget_rate, beta_reward: (0.5*forget_rate)*beta_reward,
    
    'x_V_nc': lambda forget_rate, beta_reward: (1-forget_rate)*beta_reward,
    
    'x_V_nc^2': 0,
}

mapping_x_C = {
    '1': 0,
    
    'x_C': lambda alpha_choice, beta_choice: (1-alpha_choice)*beta_choice,
    
    'c_a_repeat': lambda alpha_choice, beta_choice: (alpha_choice)*beta_choice,
    
    'x_C^2': 0,
    
    'x_C c_a_repeat': 0,
    'c_a_repeat x_C': 0,
    
    'c_a_repeat^2': 0,
}

mapping_x_C_nc = {
    '1': 0,
    
    'x_C_nc': lambda alpha_choice, beta_choice: (1-alpha_choice)*beta_choice,
    
    'x_C_nc^2': 0,
}

mappings = {
    'x_V_LR': mapping_x_V_LR,
    'x_V_nc': mapping_x_V_nc,
    'x_C': mapping_x_C,
    'x_C_nc': mapping_x_C_nc,
}

mapping_lens = (10, 3, 6, 3)

# special-cases-handles
# necessary because some sindy notations in the mappings interpet the parameters differently than AgentQ
def handle_asymmetric_learning_rates(alpha_reward, alpha_penalty):
    # in AgentQ: alpha = alpha_reward if reward > 0.5 else alpha_penalty
    # in SINDy: alpha = alpha_penalty 1 - alpha_reward r 
    if alpha_reward == 0 and alpha_penalty > 0:
        alpha_reward = alpha_penalty
    elif alpha_reward == alpha_penalty:
        alpha_reward = 0
    elif alpha_penalty == 0 and alpha_reward > 0:
        alpha_penalty = 0
        alpha_reward *= -1
    return alpha_reward, alpha_penalty

# argument extractor
def argument_extractor(data, library: str):
    if library == 'x_V_LR':
        return *handle_asymmetric_learning_rates(data['alpha_reward'], data['alpha_penalty']), data['confirmation_bias'], data['beta_reward']
    elif library == 'x_V_nc':
        return data['forget_rate'], data['beta_reward']
    elif library == 'x_C':
        return data['alpha_choice'], data['beta_choice']
    elif library == 'x_C_nc':
        return data['alpha_choice'], data['beta_choice']
    else:
        raise ValueError(f'The argument extractor for the library {library} is not implemented.')

def identified_params(true_coefs, model_coefs, library):
    # get indexes of library-mapping, where candidate terms are non-zeros
    non_zero_features, zero_features = [], []
    index_zero = []
    for feature in true_coefs:
        if not isinstance(mappings[library][feature], int):
            non_zero_features.append(feature)
        else:
            zero_features.append(feature)
    
    non_zeros_true, non_zeros_model = [], []
    for f in non_zero_features:
        non_zeros_true.append(true_coefs[f])
        non_zeros_model.append(model_coefs[f])
    non_zeros_true = np.array(non_zeros_true)
    non_zeros_model = np.array(non_zeros_model)
    
    zeros_true, zeros_model = [], []
    for f in zero_features:
        zeros_true.append(true_coefs[f])
        zeros_model.append(model_coefs[f])
    zeros_true = np.array(zeros_true)
    zeros_model = np.array(zeros_model)
    
    # for active terms
    # Check for zeros in the same location
    zeros_match = np.logical_and(non_zeros_true == 0, non_zeros_model == 0)

    # Check for non-zeros in the same location
    nonzeros_match = np.logical_and(non_zeros_true != 0, non_zeros_model != 0)

    true_pos = (np.sum(nonzeros_match)+np.sum(zeros_match))/len(non_zero_features)
    false_pos = 1 - true_pos
    
    # for non-active terms
    # Check for zeros in the same location
    true_neg = np.sum(zeros_model == 0)/len(zero_features)
    false_neg = 1 - true_neg
    
    return true_pos, true_neg, false_pos, false_neg

def n_true_params(true_coefs):
    # Count number of non-zero coefficients in AgentQ-parameters
    return np.sum([
        true_coefs['alpha_reward'] != 0,
        true_coefs['alpha_penalty'] != 0,
        true_coefs['alpha_choice'] != 0,
        true_coefs['beta_reward'] != 0,
        true_coefs['beta_choice'] != 0,
        true_coefs['forget_rate'] != 0,
        true_coefs['confirmation_bias'] != 0,
        ]).astype(int)

# plot configuration
n_params_q = 7

n_params_sindy = np.sum(mapping_lens)
bin_edges_params = np.linspace(0, n_params_q, n_params_q+1)

max_noise = 6
bin_edges_noise = max_noise-np.linspace(0, max_noise, max_noise+1)

# data configuration
base_name_data = 'data/rldm2025/data_rldm_SESSp_IT.csv'
base_name_params = 'params/rldm2025/params_rldm_SESSp_IT.pkl'

n_sessions = [16, 32, 64, 128, 256, 512]
iterations = 8                        

# identification rates
# TODO: include std in separate matrix
true_pos_params = np.zeros((len(n_sessions), n_params_q+1, len(mappings)))
true_neg_params = np.zeros((len(n_sessions), n_params_q+1, len(mappings)))
false_pos_params = np.zeros((len(n_sessions), n_params_q+1, len(mappings)))
false_neg_params = np.zeros((len(n_sessions), n_params_q+1, len(mappings)))
coefs_error_params = np.zeros((len(n_sessions), n_params_q+1, len(mappings)))

true_pos_noise = np.zeros((len(n_sessions), len(bin_edges_noise)-1, len(mappings)))
true_neg_noise = np.zeros((len(n_sessions), len(bin_edges_noise)-1, len(mappings)))
false_pos_noise = np.zeros((len(n_sessions), len(bin_edges_noise)-1, len(mappings)))
false_neg_noise = np.zeros((len(n_sessions), len(bin_edges_noise)-1, len(mappings)))
coefs_error_noise = np.zeros((len(n_sessions), len(bin_edges_noise)-1, len(mappings)))

count_model_params = np.zeros((len(n_sessions), n_params_q+1))

for index_sess, sess in enumerate(n_sessions):
    for it in range(iterations):
        # setup all sindy agents for one dataset
        rnn = base_name_params.replace('SESS', str(sess)).replace('IT', str(it))
        data = base_name_data.replace('SESS', str(sess)).replace('IT', str(it))
        agent_sindy = setup_agent_sindy(rnn, data)
        
        data = pd.read_csv(data)
        for index_participant, participant in enumerate(agent_sindy):
            # get all true parameters of current participant from dataset
            data_coefs_all = data.loc[data['session']==participant].iloc[-1]
            index_params = n_true_params(data_coefs_all)
            count_model_params[index_sess, index_params] += 1
            
            avg_noise = (data_coefs_all['beta_reward'] + data_coefs_all['beta_choice'])/2
            index_noise = np.digitize(avg_noise, bin_edges_noise, right=False)-1
            for index_library, library in enumerate(mappings):
                sindy_coefs = agent_sindy[participant]._models[library].model.steps[-1][1].coef_[0]
                # drop every entry feature that contains a u-feature (dummy-feature)
                feature_names = agent_sindy[participant]._models[library].get_feature_names()
                index_keep = ['u' not in feature for feature in feature_names]
                sindy_coefs = sindy_coefs[index_keep]
                feature_names = np.array(feature_names)[index_keep]
                sindy_coefs = {f: sindy_coefs[i] for i, f in enumerate(feature_names)}
                
                # translate data coefficient to sindy coefficients
                data_coefs = {f: 0 for f in feature_names}
                for feature in feature_names:
                    if not isinstance(mappings[library][feature], int):
                        data_coefs[feature] = mappings[library][feature](*argument_extractor(data_coefs_all, library))
                    
                # compute number of correctly identified+omitted parameters
                true_pos, true_neg, false_pos, false_neg = identified_params(data_coefs, sindy_coefs, library)
                
                true_pos_params[index_sess, index_params, index_library] += true_pos
                true_pos_noise[index_sess, index_noise, index_library] += true_pos
                true_neg_params[index_sess, index_params, index_library] += true_neg
                true_neg_noise[index_sess, index_noise, index_library] += true_neg
                false_pos_params[index_sess, index_params, index_library] += false_pos
                false_pos_noise[index_sess, index_noise, index_library] += false_pos
                false_neg_params[index_sess, index_params, index_library] += false_neg
                false_neg_noise[index_sess, index_noise, index_library] += false_neg
                
                # compute the coefficient error
                error = np.sum((np.array([data_coefs[f] - sindy_coefs[f] for f in feature_names]))**2)
                coefs_error_params[index_sess, index_params] += error
                coefs_error_noise[index_sess, index_noise] += error

# weighted average across the libraries
true_pos_params = np.average(true_pos_params, axis=-1, weights=mapping_lens/n_params_sindy)
true_pos_noise = np.average(true_pos_noise, axis=-1, weights=mapping_lens/n_params_sindy)
true_neg_params = np.average(true_neg_params, axis=-1, weights=mapping_lens / n_params_sindy)
true_neg_noise = np.average(true_neg_noise, axis=-1, weights=mapping_lens / n_params_sindy)
false_pos_params = np.average(false_pos_params, axis=-1, weights=mapping_lens / n_params_sindy)
false_pos_noise = np.average(false_pos_noise, axis=-1, weights=mapping_lens / n_params_sindy)
false_neg_params = np.average(false_neg_params, axis=-1, weights=mapping_lens / n_params_sindy)
false_neg_noise = np.average(false_neg_noise, axis=-1, weights=mapping_lens / n_params_sindy)
coefs_error_params = np.average(coefs_error_params, axis=-1, weights=mapping_lens / n_params_sindy)
coefs_error_noise = np.average(coefs_error_noise, axis=-1, weights=mapping_lens / n_params_sindy)

index_participant += 1
n_sessions_array = np.array(n_sessions).reshape(-1, 1)

# all coefficients were collected and can now be post-processed
true_pos_params = true_pos_params/count_model_params#/n_sessions_array/iterations
# true_pos_noise = true_pos_noise/count_model_params#/n_sessions_array/iterations
true_neg_params = true_neg_params/count_model_params#/n_sessions_array/iterations
# true_neg_noise = true_neg_noise/count_model_params#/n_sessions_array/iterations
false_pos_params = false_pos_params/count_model_params#/n_sessions_array/iterations
# false_pos_noise = false_pos_noise/count_model_params#/n_sessions_array/iterations
false_neg_params = false_neg_params/count_model_params#/n_sessions_array/iterations
# false_neg_noise = false_neg_noise/count_model_params#/n_sessions_array/iterations

v_min = 0
v_max = np.nanmax(np.stack((true_pos_params, false_pos_params, true_neg_params, false_neg_params)), axis=(-1, -2, -3))

coefs_error_params = coefs_error_params/n_sessions_array/n_params_sindy/iterations
# coefs_error_noise = coefs_error_noise/n_sessions_array/n_params_sindy/iterations

matrix = [
    [coefs_error_params, true_pos_params , false_pos_params], 
    [None, false_neg_params , true_neg_params], 
    # [coefs_error_noise, true_pos_noise , false_pos_noise], 
    # [None, false_neg_noise , true_neg_noise],
    ]

bin_edges_sessions = [0] + n_sessions

headers = [
    ['coef errors (MSE)', 'true positive', 'false pos'],
    [None, 'false negative', 'true negative'],
    # ['coef errors (MSE)', 'true positive', 'false pos'],
    # [None, 'false negative', 'true negative'],
]

x_axis_labels = [
    ['$n_{parameters}$', '$n_{parameters}$', '$n_{parameters}$'],
    [None, '$n_{parameters}$', '$n_{parameters}$'],
    # ['avg. noise', 'avg. noise', 'avg. noise'],
    # [None, 'avg. noise', 'avg. noise'],
]

x_axis_ticks = [
    [bin_edges_params, bin_edges_params, bin_edges_params],
    [None, bin_edges_params, bin_edges_params],
    # [bin_edges_noise, bin_edges_noise, bin_edges_noise],
    # [None, bin_edges_noise, bin_edges_noise],
]

y_axis_labels = [
    ['$n_\{sessions\}$', None, None],
    [None, '$n_\{sessions\}$', None],
    # ['$n_\{sessions\}$', None, None],
    # [None, '$n_\{sessions\}$', None],
]

# plot maps
fig, axs = plt.subplots(2, 3)
ims = []
for index_row, row in enumerate(matrix):
    for index_col, col in enumerate(row):
        if col is not None:
            im = axs[index_row, index_col].imshow(
                col, 
                origin='lower', 
                extent=[x_axis_ticks[index_row][index_col][0], 
                        x_axis_ticks[index_row][index_col][-1], 
                        bin_edges_sessions[0], 
                        bin_edges_sessions[-1],
                        ], 
                aspect='auto', 
                cmap='viridis', 
                vmin=v_min, 
                vmax=v_max,
                )
            # axs[index_row, index_col].set_title(headers[index_row][index_col])
            axs[index_row, index_col].set_xlabel(x_axis_labels[index_row][index_col])
            axs[index_row, index_col].set_ylabel(y_axis_labels[index_row][index_col])
            axs[index_row, index_col].set_yticks(n_sessions)
            if index_col == 2:
                fig.colorbar(im, ax=axs[index_row, index_col])  # Add colorbar for the first plot

plt.show()

print(true_pos_params)
print(true_neg_params)