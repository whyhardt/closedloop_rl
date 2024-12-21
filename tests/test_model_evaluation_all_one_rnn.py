import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.analysis_model_evaluation_all_one_rnn import main
from benchmarking.hierarchical_bayes_numpyro import rl_model


# data = 'data/2arm/eckstein2022_291_processed.csv'
# model = f'params/benchmarking/rnn_eckstein.pkl'
# output_file = 'benchmarking/results/results_eckstein.csv'

data = 'data/2arm/sugawara2021_143_processed.csv'
model = f'params/benchmarking/rnn_sugawara.pkl'
output_file = 'benchmarking/results/results_sugawara.csv'

main(data, model, output_file)