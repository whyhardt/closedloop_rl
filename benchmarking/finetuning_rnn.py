import sys
import os

import time
import torch
import numpy as np
import pandas as pd
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rnn_main import main as rnn_main
from utils.convert_dataset import convert_dataset
from resources.rnn_utils import DatasetRNN

def main(model, data, index: int = None, max_iterations: int = None):
    device = torch.device(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    data = convert_dataset(data)[0]
    
    # pick session if specified; else go sequentially trough dataset
    if index is not None and index > -1:
        data = DatasetRNN(data[index][0][None, :, :], data[index][1][None, :, :], device=device)
    
    for i, data_session in enumerate(data):
        if max_iterations is not None and i > max_iterations:
            break
        if index is None or index == -1:
            data_session = DatasetRNN(data_session[0][None, :, :], data_session[1][None, :, :], sequence_length=16, device=device)
        else:
            data_session = data
        rnn_main(
            model=model, 
            file_out_finetuning=model.replace('.pkl', f'_finetuned{i if index is None or index == -1 else index}.pkl'),
            checkpoint=True,
            dataset_train=data_session,
            epochs_train=0,
            epochs_finetune=1024,
            dropout=0.1,
            lr_finetune=1e-6,
            )

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Trains the RNN on behavioral data to uncover the underlying Q-Values via different cognitive mechanisms.')
    
    # Training parameters
    parser.add_argument('--model', type=str, default=None, help='Model name to load from and/or save to parameters of RNN')
    parser.add_argument('--data', type=str, default=None, help='Model name to load from parameters of RNN')
    parser.add_argument('--idx', type=int, default=-1, help='Session ID for finetuning')
    
    args = parser.parse_args()
    
    main(args.model, args.data, args.idx)