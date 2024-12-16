import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main
from resources.model_evaluation import plot_traces

model = 'ApAnBcBr'

# data = 'data/2arm/sugawara2021_143_processed.csv'
# output_file = f'benchmarking/params/sugawara2021_143/traces_test.nc'

data = 'data/2arm/eckstein2022_291_processed.csv'
output_file = f'benchmarking/params/eckstein2022_291/traces.nc'

# data = 'data/2arm/data_rnn_a05_b30_p025_ap05_varDict.csv'
# output_file = f'benchmarking/params/traces_test.nc'

mcmc = main(data, model, 2048, 512, 2, False, output_file, False)

mcmc.print_summary()
plot_traces(mcmc)