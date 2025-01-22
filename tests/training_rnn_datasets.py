import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rnn_main

path_datasets = 'data/rldm2025/'

datasets = os.listdir(path_datasets)

losses = []
for d in datasets:
    dataset = os.path.join(path_datasets, d)
    model = os.path.join('params/rldm2025', d.replace('.csv', f'.pkl').replace('data', 'params'))
    
    _, loss = rnn_main.main(
        checkpoint=False,
        epochs=512,
        
        data=dataset,
        model=model,

        n_actions=2,
        
        dropout=0.5,
        participant_emb=True,
        bagging=False,

        learning_rate=1e-2,
        batch_size=-1,
        sequence_length=64,
        train_test_ratio=0,
        n_steps_per_call=16,
        
        analysis=False,
        session_id=1,
    )

    losses.append(loss)

print(losses)