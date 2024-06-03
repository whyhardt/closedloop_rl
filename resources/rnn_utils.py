def parameter_file_naming(params_path, use_lstm, last_output, last_state, use_habit, gen_beta, forget_rate, perseverance_bias, non_binary_reward, verbose=False):
    # create name for corresponding rnn
  
    if use_lstm:
        params_path += '_lstm'
    else:
        params_path += '_rnn'
    
    if any([last_output, last_state, use_habit]):
        params_path += '_'
    
    if last_output:
        params_path += 'o'
        
    if last_state:
        params_path += 's'
        
    if use_habit:
        params_path += 'h'
    
    params_path += f'_b' + str(gen_beta).replace('.', '')
    
    if forget_rate > 0:
        params_path += f'_f' + str(forget_rate).replace('.', '')
        
    if perseverance_bias > 0:
        params_path += f'_p' + str(perseverance_bias).replace('.', '')
    
    if non_binary_reward:
        params_path += '_nonbinary'
        
    params_path += '.pkl'
    
    if verbose:
        print(f'Automatically generated name for model parameter file: {params_path}.')
        
    return params_path