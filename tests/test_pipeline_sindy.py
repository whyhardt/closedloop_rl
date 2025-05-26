import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_sindy

agent_spice, features, loss, participant_ids = pipeline_sindy.main(
    model='params/eckstein2022/rnn_eckstein2022_l1_0_0001_l2_0_0001.pkl',
    data='data/eckstein2022/eckstein2022.csv',
    save=True,
    participant_id=None,
    filter_bad_participants=True,
    use_optuna=True,
    
    # sindy parameters
    polynomial_degree=1,
    optimizer_alpha=0.1,
    optimizer_threshold=0.05,
    n_trials_off_policy=1000,
    n_sessions_off_policy=1,
    verbose=True,
    
    # generated training dataset parameters
    n_actions=2,
    sigma=0.2,
    beta_reward=1.,
    alpha=0.25,
    alpha_penalty=0.25,
    forget_rate=0.,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    analysis=True,
    get_loss=True,
)

print(f"Final loss: {loss:.4f}")
print("Kept participant IDs:", participant_ids)