import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

_, loss = rnn_main.main(
    checkpoint=True,
    epochs=0,
    
    data='data/2arm/data_rnn_br30_a025_ap05_bch30_ach05_varDict.csv',
    model='params/params_rnn_br30_a025_ap05_bch30_ach05_varDict.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    # model = f'params/benchmarking/rnn_sugawara.pkl',
    # data = 'data/2arm/sugawara2021_143_processed.csv',
    
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
    session_id=1,
)