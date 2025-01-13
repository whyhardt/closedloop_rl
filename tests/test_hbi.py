import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import main
from resources.model_evaluation import plot_traces

model = 'ApAcBcBr'

data = 'data/2arm/sugawara2021_143_processed.csv'
output_file = f'benchmarking/params/sugawara2021_143/traces_test.nc'

# data = 'data/2arm/eckstein2022_291_processed.csv'
# output_file = f'benchmarking/params/eckstein2022_291/traces.nc'

# data = 'data/2arm/data_rnn_br30_a025_ap05_bch30_ach05_varDict.csv'
# output_file = f'benchmarking/params/traces_test.nc'

mcmc = main(
    file=data, 
    model=model, 
    num_samples=4096, 
    num_warmup=1024, 
    num_chains=2, 
    hierarchical=True,
    output_file=output_file, 
    checkpoint=False,
    )

mcmc.print_summary()
plot_traces(mcmc)