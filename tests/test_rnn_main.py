import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

_, loss = rnn_main.main(
    checkpoint=True,
    epochs=0,
    
    # data='data/data_rnn_a025_b30_f02_p025_ap05_cb05_varMean.csv',
    # model='params/params_rnn_a025_b30_f02_p025_ap05_cb05_varMean_1.pkl',
    
    # model=f'params/benchmarking/rnn_eckstein.pkl',
    # data = 'data/2arm/eckstein2022_291_processed.csv',
    
    # model = f'params/benchmarking/rnn_sugawara.pkl',
    # data = 'data/2arm/sugawara2021_143_processed.csv',
    
    n_actions=2,
    
    dropout=0.1,
    bagging=True,
    # weight_decay=1e-4,

    learning_rate=1e-2,
    batch_size=-1,
    sequence_length=None,
    
    train_test_ratio=0,
    n_sessions=4*1024,
    n_trials_per_session=64,
    sigma=0.05,
    beta_reward=3.,
    alpha=0.25,
    # alpha_penalty=0.5,
    # beta_choice=3.,
    # alpha_choice=1.,
    # forget_rate=0.2,
    # confirmation_bias=0.5,
    # counterfactual=True,
    # alpha_counterfactual=0.5,
    # parameter_variance=0.,
    
    analysis=True,
)