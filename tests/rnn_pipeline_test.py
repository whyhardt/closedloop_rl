import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pipeline_rnn


_, loss = pipeline_rnn.main(
    checkpoint=False,       # For loading a checkpoint: False or Path to checkpoint
    epochs=8192, # <- 2^13 #
    
    # data='data/parameter_recovery/data_128p_0.csv',
    # model='params/parameter_recovery/rnn_128p_0.pkl',
    
    # data = 'data/optuna/data_128p_0.csv',
    # model = 'params/params_128p_0.pkl',
    
    model='params/emb-32_l2-001.pkl',
    data='data/eckstein2022/eckstein2022.csv',
    
    n_actions=2,
    
    # optuna params
    # embedding_size=22,
    # n_steps=32,
    # learning_rate=0.00023,
    
    # hand-picked params
    embedding_size=32,
    n_steps=32,
    learning_rate=1e-4,
    # l1_weight_decay=0.001,
    l2_weight_decay=0.001,#0.001,
    # dropout=0.25,
    
    # toy params for quick run
    # learning_rate=1e-3,
    # n_steps=16,
    # embedding_size=0,
    
    # other training parameters
    scheduler=True,
    train_test_ratio=0.8,
    batch_size=-1,
    sequence_length=-1,
    bagging=True,
    
    n_sessions=128,
    n_trials=200,
    sigma=0.2,
    beta_reward=1.,
    alpha_reward=0.25,
    alpha_penalty=0.25,
    forget_rate=0.3,
    confirmation_bias=0.,
    beta_choice=1.,
    alpha_choice=1.,
    counterfactual=False,
    alpha_counterfactual=0.,
    
    analysis=True,
    participant_id=64,
)