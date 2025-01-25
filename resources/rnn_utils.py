from typing import Iterable, Dict
import torch
from torch.utils.data import Dataset


class DictDataset(Dataset):
    def __init__(
        self, 
        xs: Dict[str, torch.Tensor], 
        ys: Dict[str, torch.Tensor],
        normalize_features: tuple[int] = None, 
        sequence_length: int = None,
        stride: int = 1,
        ):
        """Initializes the dataset for training the RNN. Holds information about the previous actions and rewards as well as the next action.
        Actions can be either one-hot encoded or indexes.

        Args:
            xs (torch.Tensor): Actions and rewards in the shape (n_sessions, n_timesteps, n_features)
            ys (torch.Tensor): Next action
        """
        
        self.features_xs = tuple(xs.keys())
        self.features_ys = tuple(ys.keys())
        
        # check for type of xs and ys
        for key in xs:
            if not isinstance(xs[key], torch.Tensor):
                xs[key] = torch.tensor(xs[key], dtype=torch.float32)
            if len(xs[key].shape) == 2:
                xs[key] = xs[key].unsqueeze(0)
        for key in ys:
            if not isinstance(ys[key], torch.Tensor):
                ys[key] = torch.tensor(ys[key], dtype=torch.float32)
            if len(ys[key].shape) == 2:
                ys[key] = ys[key].unsqueeze(0)
            
        # if normalize_features is not None:
        #     if isinstance(normalize_features, int):
        #         normalize_features = tuple(normalize_features)
        #     for feature in normalize_features:
        #         xs[:, :, feature] = self.normalize(xs[:, :, feature])
        
        # normalize data
        # x_std = xs.std(dim=(0, 1))
        # x_mean = xs.mean(dim=(0, 1))
        # xs = (xs - x_mean) / x_std
        # ys = (ys - x_mean) / x_std
        
        self.sequence_length = sequence_length if sequence_length is not None else ys[key].shape[1]
        self.stride = stride
        
        if sequence_length is not None:
            for key in xs:
                xs[key] = self.set_sequences(xs[key])
            for key in ys:
                ys[key] = self.set_sequences(ys[key])
        
        self.xs = xs
        self.ys = ys
        
    def normalize(self, data):
        x_min = torch.min(data)
        x_max = torch.max(data)
        return (data - x_min) / (x_max - x_min)
        
    def set_sequences(self, tensor):
        # sets sequences of length sequence_length with specified stride from the dataset
        sequences = []
        for i in range(0, max(1, tensor.shape[1]-self.sequence_length), self.stride):
            sequences.append(tensor[:, i:i+self.sequence_length, :])
        tensor = torch.cat(sequences, dim=0)
        
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(1)
            
        return tensor
    
    def __len__(self):
        return self.xs[self.features_xs[0]].shape[0]
    
    def __getitem__(self, idx):
        return {key: self.xs[key][idx, :] for key in self.xs}, {key: self.ys[key][idx, :] for key in self.ys}


def load_checkpoint(params_path, model, optimizer):
    # load trained parameters
    state_dict = torch.load(params_path, map_location=torch.device('cpu'))
    state_dict_model = state_dict['model']
    state_dict_optimizer = state_dict['optimizer']
    model.load_state_dict(state_dict_model)
    optimizer.load_state_dict(state_dict_optimizer)
    return model, optimizer


def parameter_file_naming(params_path, alpha_reward, alpha_penalty, alpha_counterfactual, confirmation_bias, forget_rate, beta_reward, alpha_choice, beta_choice, variance, verbose=False):
    # create name for corresponding rnn
  
    params_path += '_rnn'
    
    params_path += '_br' + str(beta_reward).replace('.', '')
    
    if alpha_reward > 0:
        params_path += '_a' + str(alpha_reward).replace('.', '')
    
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