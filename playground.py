import torch

state_dict = torch.load('params/benchmarking/sugawara2021_143_1.pkl')

new_state_dict = {k.replace('xH', 'xC'): v for k, v in state_dict.items()}

print(new_state_dict)

new_state_dict = {'model': new_state_dict,
                  'optimizer': None}

torch.save(new_state_dict, 'params/benchmarking/sugawara2021_143_1.pkl')