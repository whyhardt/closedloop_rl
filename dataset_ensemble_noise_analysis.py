import sys

from rnn_main import main

if __name__ == '__main__':
  
    row_params_input = int(sys.argv[1])

    params_input = []
    with open('repositories/closedloop_rl/analysis/dataset_ensemble_noise_analysis.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == row_params_input:
                n_sessions = int(line.split(',')[0])
                n_submodels = int(line.split(',')[1])
                beta = int(line.split(',')[2])
    
    print(f'n_sessions = {n_sessions}')
    print(f'n_submodels = {n_submodels}')
    print(f'beta = {beta}')
    model = f'params/params_rnn_sessions{n_sessions}_submodels{n_submodels}_noise{beta}'
    
    main(
        checkpoint = False,
        model = model,

        # training parameters
        epochs = 1024,
        n_trials_per_session = 64,
        n_sessions = n_sessions,
        n_steps_per_call = 8,
        bagging = True,
        learning_rate = 1e-2,

        # ensemble parameters
        n_submodels = n_submodels,
        ensemble = 1,
        
        # rnn parameters
        hidden_size = 8,
        dropout = 0.1,
        
        # ground truth parameters
        alpha = 0.25,
        beta = beta,
        forget_rate = 0.2,
        perseveration_bias = 0.25,
        alpha_penalty = True,
        confirmation_bias = True,
        # reward_update_rule = lambda q, reward: reward-q,
        
        # environment parameters
        sigma = 0.1,
        
        analysis = False,
    )