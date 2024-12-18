from rnn_main import main

if __name__ == '__main__':
  
    row_n_steps_per_call = 1#int(sys.argv[1])
    
    # n_steps_per_call = np.loadtxt('analysis/n_steps_per_call_analysis.csv', delimiter=',', )[0].to_numpy()[row_n_steps_per_call]
    
    n_steps_per_call = []
    with open('analysis/n_steps_per_call_analysis.csv', 'r') as f:
        for line in f:
            n_steps_per_call.append(int(line.split(',')[0])) 
    n_steps_per_call = n_steps_per_call[row_n_steps_per_call]
    
    print(f'n_steps_per_call = {n_steps_per_call}')
    model = f'params/params_rnn_steps{n_steps_per_call}'
    
    main(
        checkpoint = False,
        model = model,

        # training parameters
        epochs = 128,
        n_trials_per_session = 64,
        n_sessions = 4096,
        n_steps_per_call = n_steps_per_call,
        bagging = True,
        learning_rate = 1e-2,

        # ensemble parameters
        n_submodels = 8,
        ensemble = 1,
        
        # rnn parameters
        hidden_size = 8,
        dropout = 0.1,
        
        # ground truth parameters
        alpha = 0.25,
        beta_reward = 3,
        forget_rate = 0.2,
        beta_choice = 0.25,
        alpha_penalty = True,
        confirmation_bias = True,
        # reward_update_rule = lambda q, reward: reward-q,
        
        # environment parameters
        sigma = 0.1,
        
        analysis = False,
    )