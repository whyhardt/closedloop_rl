import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sindy_main

sindy_main.main(
    # sindy parameters
    polynomial_degree=2,
    threshold=0.05,
    regularization=0,
    
    # generated training dataset parameters
    n_trials_per_session = 200,
    n_sessions = 100,
    
    alpha=0.25,
    regret=False,
    confirmation_bias=False,
    forget_rate=0.,
    perseveration_bias=0.25,
    beta=5,
    directed_exploration_bias=1.,
    undirected_exploration_bias=5.,
    
    analysis=True,
)