import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy


def test_original_optimizer():
    print("\n Testing original optimizer")
    try:
        agent_spice, features, loss = pipeline_sindy.main(
            model = 'params/eckstein2022/params_eckstein2022.pkl',
            data = 'data/eckstein2022/eckstein2022.csv',
            participant_id=1,
            polynomial_degree=1,
            optimizer_type="original",  # when we want toUse the original implementation
            optimizer_alpha=0.1,
            optimizer_threshold=0.05,
            n_trials_off_policy=1024,
            n_sessions_off_policy=0,
            verbose=True,
            analysis=True,
            get_loss=True,
        )
        
        if agent_spice is None:
            print("Error: Failed to create SPICE agent with original optimizer")
            return None
            
        print(f"Loss with original optimizer: {loss}")
        return loss
    except Exception as e:
        print(f"Error in test_original_optimizer: {str(e)}")
        return None


# Using a specific optimizer type (STLSQ)
def test_specific_optimizer():
    print("\nTesting STLSQ optimizer")
    try:
        agent_spice, features, loss = pipeline_sindy.main(
            model = 'params/eckstein2022/params_eckstein2022.pkl',
            data = 'data/eckstein2022/eckstein2022.csv',
            participant_id=1,
            polynomial_degree=1,
            optimizer_type="STLSQ",  # Specify a particular optimizer
            optimizer_alpha=0.1,
            optimizer_threshold=0.05,
            n_trials_off_policy=1024,
            n_sessions_off_policy=0,
            verbose=True,
            analysis=True,
            get_loss=True,
        )
        
        if agent_spice is None:
            print("Error: Failed to create SPICE agent with STLSQ optimizer")
            return None
            
        print(f"Loss with STLSQ optimizer: {loss}")
        return loss
    except Exception as e:
        print(f"Error in test_specific_optimizer: {str(e)}")
        return None


#  Using automatic optimizer selection
def test_auto_optimizer():
    print("\nTesting automatic optimizer selection ")
    try:
        agent_spice, features, loss = pipeline_sindy.main(
            model = 'params/eckstein2022/params_eckstein2022.pkl',
            data = 'data/eckstein2022/eckstein2022.csv',
            participant_id=1,
            polynomial_degree=1,
            optimizer_type="auto",  # Let Optuna find the best optimizer
            optuna_trials=20,       # Try 20 different combinations
            optuna_timeout=300,     # ? 5-minute timeout
            n_trials_off_policy=1024,
            n_sessions_off_policy=0,
            verbose=True,
            analysis=True,
            get_loss=True,
        )
        
        if agent_spice is None:
            print("Error: Failed to create SPICE agent with auto optimizer")
            return None
            
        print(f"Loss with automatic optimizer selection: {loss}")
        return loss
    except Exception as e:
        print(f"Error in test_auto_optimizer: {str(e)}")
        return None


# Run all tests
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SINDy optimizer selection")
    parser.add_argument("--run_all", action="store_true", help="Run all tests")
    parser.add_argument("--original", action="store_true", help="Test original optimizer")
    parser.add_argument("--specific", action="store_true", help="Test specific optimizer")
    parser.add_argument("--auto", action="store_true", help="Test automatic optimizer selection")
    args = parser.parse_args()
    
    if args.run_all or (not args.original and not args.specific and not args.auto):
        # Run all tests if --run_all is specified or no specific test is selected
        loss_original = test_original_optimizer()
        loss_specific = test_specific_optimizer()
        loss_auto = test_auto_optimizer()
        
        print("\nSummary ")
        results = []
        
        if loss_original is not None:
            print(f"Original optimizer loss: {loss_original}")
            results.append(("Original", loss_original))
            
        if loss_specific is not None:
            print(f"STLSQ optimizer loss: {loss_specific}")
            results.append(("STLSQ", loss_specific))
            
        if loss_auto is not None:
            print(f"Auto-selected optimizer loss: {loss_auto}")
            results.append(("Auto", loss_auto))
        
        if results:
            best_optimizer, best_loss = min(results, key=lambda x: x[1])
            print(f"\n{best_optimizer} optimizer performed best with loss: {best_loss}")
        else:
            print("\nNo successful tests to compare.")
    else:
        if args.original:
            test_original_optimizer()
        if args.specific:
            test_specific_optimizer()
        if args.auto:
            test_auto_optimizer()