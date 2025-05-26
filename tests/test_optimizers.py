import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy
from resources.rnn_utils import DatasetRNN

def test_optuna_optimization():
    print("\nTesting per-participant optimizer optimization with Optuna")
    try:
        agent_spice, features, loss = pipeline_sindy.main(
            model='params/eckstein2022/params_eckstein2022.pkl',
            data='data/eckstein2022/eckstein2022.csv',
            participant_id=7, #None,  # Process all participants
            polynomial_degree=1,
            n_trials_off_policy=1024,
            n_sessions_off_policy=0,
            verbose=True,
            analysis=True,
            get_loss=True,
            use_optuna=True,  # Enable Optuna optimization
            filter_bad_participants=True,  # filter out 
            #show_plots=True,  # show plots
        )
        
        if agent_spice is None:
            print("Error: Failed to create SPICE agent with Optuna optimization")
            return None
            
        print(f"Loss with per-participant optimizer optimization: {loss}")
        return loss
    except Exception as e:
        print(f"Error in test_optuna_optimization: {str(e)}")
        return None

if __name__ == "__main__":
    test_optuna_optimization()