# Add resources folder to path
from rnn import BaseRNN, EnsembleRNN


def parameter_file_naming(params_path, use_lstm, last_output, last_state, gen_beta, forget_rate, perseverance_bias, non_binary_reward, verbose=False):
    # create name for corresponding rnn
  
    if use_lstm:
        params_path += '_lstm'
    else:
        params_path += '_rnn'
    
    if any([last_output, last_state]):
        params_path += '_'
    
    if last_output:
        params_path += 'o'
        
    if last_state:
        params_path += 's'
    
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


def check_ensemble_rnn():
    """
    Checks that EnsembleRNN class is correctly implemented i.e. that it has the same methods as the BaseRNN class. 
    BaseRNN.forward should be matched with EnsembleRNN.__call__.
    """
    
    import inspect
    
    list_missing_methods = []
    list_exceptions = ['append_timestep_sample']
    
    # Get all methods of the class
    methods_ensemble_rnn = inspect.getmembers(EnsembleRNN, predicate=inspect.isfunction)
    methods_base_rnn = inspect.getmembers(BaseRNN, predicate=inspect.isfunction)
    # Filter out methods that are not defined in this class
    methods_ensemble_rnn = [name for name, func in methods_ensemble_rnn if func.__qualname__.startswith(EnsembleRNN.__name__ + '.')]
    methods_base_rnn = [name for name, func in methods_base_rnn if func.__qualname__.startswith(BaseRNN.__name__ + '.')]
    # Check if all methods of BaseRNN are implemented in EnsembleRNN
    # match forward with __call__
    if 'forward' in methods_base_rnn and '__call__' in methods_ensemble_rnn:
        pass
    else:
        list_missing_methods.append('forward')
    methods_base_rnn.remove('forward')
    methods_ensemble_rnn.remove('__call__')
    # Filter all methods from EnsembleRNN that are not in BaseRNN
    methods_ensemble_rnn = [name for name in methods_ensemble_rnn if name in methods_base_rnn]
    for method in methods_base_rnn:
        if method not in methods_ensemble_rnn and method not in list_exceptions:
            list_missing_methods.append(method)
    
    if len(list_missing_methods) == 0:
        print('EnsembleRNN is correctly implemented.')
    else:
        print('EnsembleRNN is not correctly implemented. The following methods are missing:')
        print(list_missing_methods)