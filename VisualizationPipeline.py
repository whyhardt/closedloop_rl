import sys
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import numpy as np

from sklearn.decomposition import PCA
import pysindy as ps

warnings.filterwarnings("ignore")

# RL libraries
sys.path.append('resources')  # add source directoy to path
from resources import bandits

from labrotation.sampling import generate_libraries, sample_custom_library
from labrotation.agents import dict_agents, get_q, AgentSindy, make_sindy_data

from utils.interactive_plots import interactive_scatter_3dplot


MODEL_NUM = 1000

# experiment parameters
n_trials_per_session = 200
n_sessions = 220

# environment setup
non_binary_reward = False
n_actions = 2
sigma = .1
environment = bandits.EnvironmentBanditsDrift(sigma=sigma, n_actions=n_actions, non_binary_rewards=non_binary_reward)
agent = bandits.AgentQ(0.25, 5, n_actions, 0, 0) 
dataset_test, experiment_list_test = bandits.create_dataset(agent=agent,environment=environment,n_trials_per_session=n_trials_per_session,n_sessions=1)

# SINDy library
custom_lib_functions = [
    # sub-library which is always included
    lambda q, c, r: 1,
    lambda q, c, r: q,
    lambda q, c, r: r,
    lambda q, c, r: np.power(q, 2),
    lambda q, c, r: q * r,
    lambda q, c, r: np.power(r, 2),
    # sub-library if the possible action was chosen
    lambda q, c, r: c,
    lambda q, c, r: c * q,
    lambda q, c, r: c * r,
    lambda q, c, r: c * np.power(q, 2),
    lambda q, c, r: c * q * r,
    lambda q, c, r: c * np.power(r, 2),
]

custom_lib_names = [
    # sub-library which is always included
    '1',
    'q',
    'r',
    'q^2',
    'q*r',
    'r^2',
    # sub-library if the possible action was chosen
    'c',
    'c*q',
    'c*r',
    'c*q^2',
    'c*q*r',
    'c*r^2',
]

# make SINDy library
custom_lib = ps.CustomLibrary(
    custom_lib_functions,
    custom_lib_names,
)

# get max length of the library names
max_lib_name_len = max([len(name) for name in custom_lib_names])
# make all names the same length by adding spaces
custom_lib_names_plot = [name + " " * (max_lib_name_len - len(name) + 3) for name in custom_lib_names]
# set more strings for plot
str_terms = "Terms:"
str_sampled = "Sampled:"
str_recovered = "Recovered:"
# make all names the same length by adding spaces
max_str_len = max([len(str_terms), len(str_sampled), len(str_recovered)])
str_terms += " " * (max_str_len - len(str_terms) + 3)
str_sampled += " " * (max_str_len - len(str_sampled) + 3)
str_recovered += " " * (max_str_len - len(str_recovered) + 3)
# get cell width for the table
cell_width = max([max_lib_name_len + 3, max_str_len + 3])
# make all lib_names and strings the same length by adding spaces
custom_lib_names_plot = [name + " " * (cell_width - len(name)) for name in custom_lib_names]
str_terms += " " * (cell_width - len(str_terms))
str_sampled += " " * (cell_width - len(str_sampled))
str_recovered += " " * (cell_width - len(str_recovered))
                        
# sindy hyperparameter
threshold = 0.01
dt = 1

# sampling hyperparameters
rng = np.random.default_rng()

# loop for sampling random models, fitting SINDy and computing the mean squared error between the coefficients
qs = []
mse_models = []
popup_plotly = []
count_errors = 0
from tqdm import tqdm
for i in tqdm(range(MODEL_NUM), ascii=True, desc="Sampling models"):
    # sample random model
    lib, lib_names, w, mask = sample_custom_library(custom_lib_functions, custom_lib_names, rng.laplace)
    
    # define update rule with global variables: lib, w
    # update function for sampled model
    def update_function_sampled(q, c, r):
        return np.sum([l(q, c, r) * w_l for l, w_l in zip(lib, w)])
    
    # setup a SINDy model with the sampled library weights
    sampled_groundtruth = AgentSindy(alpha=0, beta=5, n_actions=n_actions)
    sampled_groundtruth.set_update_rule(update_function_sampled)
    
    try:
        # generate data for the sampled model
        dataset_sampled, experiment_list_sampled = bandits.create_dataset(
            agent=sampled_groundtruth,
            environment=environment,
            n_trials_per_session=n_trials_per_session,
            n_sessions=10
            )
    except ValueError as e:
        # print("Error in creating dataset (Probably NaN for choice_probs). Continue with next library.")
        count_errors += 1
        continue
    
    x_train, control, feature_names = make_sindy_data(experiment_list_sampled, sampled_groundtruth, get_choices=True)
    # scale q-values between 0 and 1
    x_max = np.max(np.stack(x_train, axis=0))
    x_min = np.min(np.stack(x_train, axis=0))
    x_train = [(x - x_min) / (x_max - x_min) for x in x_train]
    # replace NaN values with 0
    x_train = [np.nan_to_num(x) for x in x_train]
    
    # fit a SINDy model to the data of the sampled model
    datasindy = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold, verbose=False, alpha=0.1),
        feature_library=custom_lib,
        discrete_time=True,
        feature_names=feature_names,
    )
    datasindy.fit(x_train, t=dt, u=control, ensemble=False, library_ensemble=False, multiple_trajectories=True, quiet=True)
    # datasindy.print()
    # set new sindy update rule and synthesize new dataset
    def update_rule_sindy(q, choice, reward):
        return datasindy.simulate(q, t=2, u=np.array([choice, reward]).reshape(1, 2))[-1]
    datasindyagent = AgentSindy(alpha=0, beta=5, n_actions=n_actions)
    datasindyagent.set_update_rule(update_rule_sindy, datasindy.equations)

    # get q-values for a specific trajectory for the sampled model for later PCA
    q = get_q(experiment_list_test[0], sampled_groundtruth)[0]
    # scale q-values between 0 and 1
    x_max = np.max(np.stack(q, axis=0))
    x_min = np.min(np.stack(q, axis=0))
    q = [(x - x_min) / (x_max - x_min) for x in q]
    # replace NaN values with 0
    q = [np.nan_to_num(x) for x in q]
    qs.append(q)

    # compute the mean squared error between the sampled model coefficients and the SINDy model coefficients
    sindy_coeffs = datasindy.coefficients()
    sampled_coeffs = np.zeros_like(sindy_coeffs)
    sampled_coeffs[0, mask] = w
    mse_models.append(np.mean((sindy_coeffs - sampled_coeffs) ** 2))
    
    # add the text for the pop-up textbox in plotly
    # make a three liner 
    # first line: mse = ...
    # second and third line: table of coefficients with the sampled and recovered ones
    # text = f"MSE = {mse_models[-1]:.3f}<br>"
    # text += "<table>"
    # for i, (sindy_coeff, sampled_coeff) in enumerate(zip(sindy_coeffs.T, sampled_coeffs.T)):
    #     text += f"<tr><td>{custom_lib_names[i]}</td><td>{sampled_coeff[0]:.3f}</td><td>{sindy_coeff[0]:.3f}</td></tr>"
    # text += "</table>"
    text = f"MSE = {mse_models[-1]:.3f}<br>"
    # text += "Coefficients:<br>"
    line_lib_functions = ""
    line_sampled_coeffs = ""
    line_sindy_coeffs = ""
    
    for i, (sindy_coeff, sampled_coeff) in enumerate(zip(sindy_coeffs.T, sampled_coeffs.T)):
        # add a line with all library functions
        line_lib_functions += f"{custom_lib_names_plot[i]}"
        # add a line with all sampled coefficients
        line_sampled_coeffs += f"{sampled_coeff[0]:.3f}" + " " * (cell_width - 6)
        # add a line with all sindy coefficients
        line_sindy_coeffs += f"{sindy_coeff[0]:.3f}" + " " * (cell_width - 6)
    text += f"{str_terms}{line_lib_functions}<br>"
    text += f"{str_sampled}{line_sampled_coeffs}<br>"
    text += f"{str_recovered}{line_sindy_coeffs}"
    popup_plotly.append(text)
    
# filter absurdly high mse values
print(f"Number of models before filtering: {len(qs)}")
mse_models_old = mse_models

threshold_mse = 0.5
mse_models = np.array(mse_models)
mask = mse_models < threshold_mse
qs = np.array(qs)[mask]
mse_models = mse_models[mask]

print(f"Number of models after filtering: {len(qs)}")

# perform PCA
pca = PCA(3)
qs_array = np.stack([q_model[:, 0] for q_model in qs], axis=0)
res = pca.fit_transform(qs_array)

# plot the PCA
interactive_scatter_3dplot(res[:, 0], res[:, 1], res[:, 2], mse_models, "3D PCA with MSE", marker_text=popup_plotly)