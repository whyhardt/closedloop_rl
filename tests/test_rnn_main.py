import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

_, loss = rnn_main.main(
    checkpoint=True,
    epochs=0,
    
    # data='data/rldm2025/data_rldm_128p_5.csv',
    # model='params/rldm2025/params_rldm_128p_5.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    model = f'params/benchmarking/rnn_sugawara.pkl',
    data = 'data/2arm/sugawara2021_143_processed.csv',
    
    n_actions=2,
    
    dropout=0.5,
    participant_emb=True,
    bagging=False,

    learning_rate=1e-2,
    batch_size=-1,
    sequence_length=None,
    train_test_ratio=0,
    n_steps_per_call=16,
    
    # n_sessions=512,
    # n_trials_per_session=256,
    # sigma=0.2,
    # beta_reward=3.,
    # alpha=0.25,
    # alpha_penalty=0.5,
    # beta_choice=3.,
    # alpha_choice=1.,
    # forget_rate=0.2,
    # confirmation_bias=0.5,
    # counterfactual=True,
    # alpha_counterfactual=0.5,
    # parameter_variance=0.,
    
    analysis=True,
    session_id=0,
)