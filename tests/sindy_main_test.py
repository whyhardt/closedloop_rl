import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sindy_main

sindy_main.main(
    model = 'params/sugawara2021_143.pkl',
    data = 'data/sugawara2021_143_processed.csv',
    
    # sindy parameters
    polynomial_degree=2,
    threshold=0.05,
    regularization=0,
    
    # generated training dataset parameters
    n_trials_per_session = 256,
    n_sessions = 128,
    
    # alpha=0.25,
    # regret=True,
    # confirmation_bias=True,
    # forget_rate=0.2,
    # perseveration_bias=0.25,
    
    analysis=True,
)