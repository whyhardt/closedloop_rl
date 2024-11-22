import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.finetuning_rnn import main

model = 'params/benchmarking/rnn_sugawara2021_143_1.pkl'
data = 'data/sugawara2021_143_processed.csv'

main(model, data, -1)