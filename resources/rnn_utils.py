import torch
from torch.utils.data import Dataset

# Add resources folder to path
from rnn import BaseRNN, EnsembleRNN


class DatasetRNN(Dataset):
    def __init__(
        self, 
        xs: torch.Tensor, 
        ys: torch.Tensor, 
        sequence_length: int = None,
        stride: int = 1,
        device=torch.device('cpu'),
        ):
        """Initializes the dataset for training the RNN. Holds information about the previous actions and rewards as well as the next action.
        Actions can be either one-hot encoded or indexes.

        Args:
            xs (torch.Tensor): Actions and rewards in the shape (n_sessions, n_timesteps, n_features)
            ys (torch.Tensor): Next action
            batch_size (Optional[int], optional): Sets batch size if desired else uses n_samples as batch size.
            device (torch.Device, optional): Torch device. If None, uses cuda if available else cpu.
        """
        
        # check for type of xs and ys
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs, dtype=torch.float32)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys, dtype=torch.float32)
            
        # check dimensions of xs and ys
        if len(xs.shape) == 2:
            xs = xs.unsqueeze(0)
        if len(ys.shape) == 2:
            ys = ys.unsqueeze(0)
        
        self.sequence_length = sequence_length if sequence_length is not None else xs.shape[1]
        self.stride = stride
        
        if sequence_length is not None:
            xs, ys = self.set_sequences(xs, ys)
        
        self.xs = xs.to(device)
        self.ys = ys.to(device)
        
    def set_sequences(self, xs, ys):
        # sets sequences of length sequence_length with specified stride from the dataset
        xs_sequences = []
        ys_sequences = []
        for i in range(0, max(1, xs.shape[1]-self.sequence_length), self.stride):
            xs_sequences.append(xs[:, i:i+self.sequence_length, :])
            ys_sequences.append(ys[:, i:i+self.sequence_length, :])
        xs = torch.cat(xs_sequences, dim=0)
        ys = torch.cat(ys_sequences, dim=0)
        
        if len(xs.shape) == 2:
            xs = xs.unsqueeze(1)
            ys = ys.unsqueeze(1)
            
        return xs, ys
    
    def __len__(self):
        return self.xs.shape[0]
    
    def __getitem__(self, idx):
        return self.xs[idx, :], self.ys[idx, :]


def parameter_file_naming(params_path, use_lstm, last_output, last_state, gen_beta, forget_rate, perseverance_bias, correlated_reward, non_binary_reward, verbose=False):
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
        
    if correlated_reward:
        params_path += '_c'
    
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