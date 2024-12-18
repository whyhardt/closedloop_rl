import torch
from torch.utils.data import Dataset

# Add resources folder to path
from resources.rnn import BaseRNN, EnsembleRNN


class DatasetRNN(Dataset):
    def __init__(
        self, 
        xs: torch.Tensor, 
        ys: torch.Tensor,
        normalize_features: tuple[int] = None, 
        sequence_length: int = None,
        stride: int = 1,
        device=None,
        ):
        """Initializes the dataset for training the RNN. Holds information about the previous actions and rewards as well as the next action.
        Actions can be either one-hot encoded or indexes.

        Args:
            xs (torch.Tensor): Actions and rewards in the shape (n_sessions, n_timesteps, n_features)
            ys (torch.Tensor): Next action
            batch_size (Optional[int], optional): Sets batch size if desired else uses n_samples as batch size.
            device (torch.Device, optional): Torch device. If None, uses cuda if available else cpu.
        """
        
        if device is None:
            device = torch.device('cpu')
        
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
            
        if normalize_features is not None:
            if isinstance(normalize_features, int):
                normalize_features = tuple(normalize_features)
            for feature in normalize_features:
                xs[:, :, feature] = self.normalize(xs[:, :, feature])
        
        # normalize data
        # x_std = xs.std(dim=(0, 1))
        # x_mean = xs.mean(dim=(0, 1))
        # xs = (xs - x_mean) / x_std
        # ys = (ys - x_mean) / x_std
        
        self.sequence_length = sequence_length if sequence_length is not None else xs.shape[1]
        self.stride = stride
        
        if sequence_length is not None:
            xs, ys = self.set_sequences(xs, ys)
        self.device = device
        self.xs = xs
        self.ys = ys
        
    def normalize(self, data):
        x_min = torch.min(data)
        x_max = torch.max(data)
        return (data - x_min) / (x_max - x_min)
        
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


def load_checkpoint(params_path, model, optimizer, voting_type=None):
    # load trained parameters
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))
    state_dict_model = state_dict['model']
    state_dict_optimizer = state_dict['optimizer']
    if isinstance(state_dict_model, dict):
      for m, o in zip(model, optimizer):
        m.load_state_dict(state_dict_model)
        o.load_state_dict(state_dict_optimizer)
    elif isinstance(state_dict_model, list):
        print('Loading ensemble model...')
        for i, state_dict_model_i, state_dict_optim_i in zip(len(state_dict_model), state_dict_model, state_dict_optimizer):
            model[i].load_state_dict(state_dict_model_i)
            optimizer[i].load_state_dict(state_dict_optim_i)
        model = EnsembleRNN(model, voting_type=voting_type)
    return model, optimizer


def parameter_file_naming(params_path, alpha, alpha_penalty, alpha_counterfactual, confirmation_bias, forget_rate, beta_reward, alpha_choice, beta_choice, variance, verbose=False):
    # create name for corresponding rnn
  
    params_path += '_rnn'
    
    params_path += '_br' + str(beta_reward).replace('.', '')
    
    if alpha > 0:
        params_path += '_a' + str(alpha).replace('.', '')
    
    if alpha_penalty >= 0:
        params_path += '_ap' + str(alpha_penalty).replace('.', '')
    
    if alpha_counterfactual > 0:
        params_path += '_ac' + str(alpha_counterfactual).replace('.', '')
        
    if beta_choice > 0 and alpha_choice > 0:
        params_path += '_bch' + str(beta_choice).replace('.', '')
        params_path += '_ach' + str(alpha_choice).replace('.', '')
    
    if forget_rate > 0:
        params_path += '_f' + str(forget_rate).replace('.', '')
        
    if confirmation_bias > 0:
        params_path += '_cb' + str(confirmation_bias).replace('.', '')
    
    if not isinstance(variance, dict):
        if variance != 0:
            params_path += '_var' + str(variance).replace('.', '').replace('-1','Mean')
    else:
        params_path += '_varDict'
        
    # if non_binary_reward:
    #     params_path += '_nonbinary'
        
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