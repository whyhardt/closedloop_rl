import sys, os
import numpyro
import pickle
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmarking.hierarchical_bayes_numpyro import rl_model


def main(model):
    # get summary
    with open(model, 'rb') as file:
        mcmc = pickle.load(file)
    mcmc.print_summary()
    
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Get summary from numpyro model.')
    parser.add_argument('--model', type=str, help='numpyro rl model')
    args = parser.parse_args()
    main(args.model)
