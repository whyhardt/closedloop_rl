import pickle

import numpy as np
import pysindy as ps

from resources import bandits, hybrnn_forget
from resources.datasets import SyntheticDataset
from resources.models import AgentQuadQ, AgentNetwork_VisibleState
from resources.trainers import RNNTrainer, SindyTrainer, RNNSindyEstimator


def main(forgetting_rate=0.1, perseveration_bias=0.0, n_sessions=220, n_trials_per_session=200, sigma=0.1,
         non_binary_reward=False,
         params_path='rnn_params.pkl', agent_kw='basic', train=True, load=False,
         loss_function='categorical',
         convergence_thresh=1e-5, n_steps_max=10000,
         habit_weight=0.0, n_actions=2, get_choices=False, threshold=0.01,
         ensemble=False, library_ensemble=False,
         alpha=0.1, beta=3, dt=1.0, normalize=False, verbose=False):
    dict_agents = {
        'basic': lambda alpha, beta, n_actions, forgetting_rate, perseveration_bias: bandits.AgentQ(alpha, beta,
                                                                                                    n_actions,
                                                                                                    forgetting_rate,
                                                                                                    perseveration_bias),
        'quad_q': lambda alpha, beta, n_actions, forgetting_rate, perseveration_bias: AgentQuadQ(alpha, beta, n_actions,
                                                                                                 forgetting_rate,
                                                                                                 perseveration_bias)
    }
    # create dataset
    agent = dict_agents[agent_kw](alpha, beta, n_actions, forgetting_rate, perseveration_bias)
    dataset = SyntheticDataset(agent, n_sessions, n_trials_per_session, sigma, non_binary_reward)

    # train sindy
    # library = create_custom_library()
    poly_order = 3
    library = ps.PolynomialLibrary(poly_order)

    rnn_fun = get_rnn_maker(habit_weight, n_actions)

    rnn_sindy_estimator = RNNSindyEstimator(
        params_path=params_path,
        rnn_fun=rnn_fun,
        train=train,
        load=load,
        loss_function=loss_function,
        n_trials_per_session=n_trials_per_session,
        n_sessions=n_sessions,
        convergence_thresh=convergence_thresh,
        n_steps_max=n_steps_max,
        habit_weight=habit_weight,
        n_actions=n_actions,
        agent=agent,
        get_choices=get_choices,
        threshold=threshold,
        library=library,
        ensemble=ensemble,
        library_ensemble=library_ensemble,
        non_binary_reward=non_binary_reward,
        alpha=alpha,
        beta=beta,
        sigma=sigma,
        dt=dt,
        normalize=normalize,
        verbose=verbose
    )
    rnn_sindy_estimator.fit((dataset.dataset_train, dataset.experiment_list_train))

    return rnn_sindy_estimator, dataset, agent


def get_rnn_maker(habit_weight=0.0, n_actions=2):
    # Is the model recurrent (ie can it see the hidden state from the previous step)
    use_hidden_state = False
    # Is the model recurrent (ie can it see the hidden state from the previous step)
    use_previous_values = False  # @param ['True', 'False']
    # If True, learn a value for the forgetting term
    fit_forget = False
    value_weight = 1.  # This is needed for it to be doing RL
    rnn_rl_params = {
        's': use_hidden_state,
        'o': use_previous_values,
        'fit_forget': fit_forget,
        'forget': 0.,
        'w_h': habit_weight,
        'w_v': value_weight}
    network_params = {'n_actions': n_actions, 'hidden_size': 16}
    return lambda: hybrnn_forget.BiRNN(rl_params=rnn_rl_params, network_params=network_params)


def create_custom_library():
    custom_lib_functions = [
        # sub-library which is always included
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
        # part library which is always included
        lambda q, c, r: f'{q}',
        lambda q, c, r: f'{r}',
        lambda q, c, r: f'{q}^2',
        lambda q, c, r: f'{q}*{r}',
        lambda q, c, r: f'{r}^2',
        # part library if the possible action was chosen
        lambda q, c, r: f'{c}',
        lambda q, c, r: f'{c}*{q}',
        lambda q, c, r: f'{c}*{r}',
        lambda q, c, r: f'{c}*{q}^2',
        lambda q, c, r: f'{c}*{q}*{r}',
        lambda q, c, r: f'{c}*{r}^2',
    ]
    return ps.CustomLibrary(
        library_functions=custom_lib_functions,
        function_names=custom_lib_names,
        include_bias=True,
    )


if __name__ == '__main__':
    main(forgetting_rate=0.1, perseveration_bias=0.0, n_sessions=220, n_trials_per_session=200, sigma=0.1,
         non_binary_reward=False,
         params_path='rnn_params.pkl', agent_kw='basic', train=True, load=False,
         loss_function='categorical',
         convergence_thresh=1e-5, n_steps_max=10000,
         habit_weight=0.0, n_actions=2, get_choices=True, threshold=0.01,
         ensemble=False, library_ensemble=False,
         alpha=0.1, beta=3, dt=1.0, normalize=False, verbose=False)
