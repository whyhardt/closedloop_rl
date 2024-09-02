import os
import re
import matplotlib.pyplot as plt
import numpy as np

from rnn_main import main as rnn_main
from resources import bandits

if __name__=='__main__':
    # directory holding all the models
    path = 'params/dataset_ensemble_noise_analysis_run1/'
    
    # marker dictionary based on n_submodels
    markers = {
        16: 'o',
        8: 'd',
        1: 'x',
    }
    
    colors = {
        5: 'tab:green',
        3: 'tab:orange',
        1: 'tab:red',
    }
    
    x_ticks = {
        4: 0,
        8: 1,
        32: 2,
        64: 3,
        128: 4,
        256: 5,
        512: 6,
        1024: 7,
        4096: 8,
    }
    
    print('Creating the test dataset...', end='\r')
    environment = bandits.EnvironmentBanditsDrift(sigma=0.1)
    agent_b5 = bandits.AgentQ(alpha=0.25, beta=5, forget_rate=0.2, perseveration_bias=0.25, regret=True, confirmation_bias=True)
    agent_b3 = bandits.AgentQ(alpha=0.25, beta=3, forget_rate=0.2, perseveration_bias=0.25, regret=True, confirmation_bias=True)
    agent_b1 = bandits.AgentQ(alpha=0.25, beta=1, forget_rate=0.2, perseveration_bias=0.25, regret=True, confirmation_bias=True)
    datasets = {
        5: bandits.create_dataset(agent=agent_b5, environment=environment, n_trials_per_session=200, n_sessions=1024),
        3: bandits.create_dataset(agent=agent_b3, environment=environment, n_trials_per_session=200, n_sessions=1024),
        1: bandits.create_dataset(agent=agent_b1, environment=environment, n_trials_per_session=200, n_sessions=1024)
    }
    
    # get a list of model 
    files = os.listdir(path)
    losses = []
    for f in files:
        # get all numbers from string
        numbers = re.findall(r'\d+', f)
        n_sessions = int(numbers[0])
        n_submodels = int(numbers[1])
        beta = int(numbers[2])

        losses.append(rnn_main(
            checkpoint=True,
            model=path+f,
            hidden_size=8,
            n_submodels=n_submodels,
            n_sessions=n_sessions,
            ensemble=1,
            epochs=0,
            alpha=0.25,
            beta=beta,
            forget_rate=0.2,
            perseveration_bias=0.25,
            regret=True,
            confirmation_bias=True,
            sigma=0.1,
            dataset_test=datasets[beta][0],
            experiment_list_test=datasets[beta][1],
            )
        )
        
        if n_submodels == 16:
            pos = 0.2
        elif n_submodels == 8:
            pos = 0
        elif n_submodels == 1:
            pos = -0.2
        
        plt.scatter(x_ticks[n_sessions] + pos, losses[-1], c=colors[beta], marker=markers[n_submodels])
    plt.ylabel('Loss')
    plt.xlabel('Sessions')
    plt.xticks(np.arange(0, len(x_ticks)), list(x_ticks.keys()))
    plt.show()
        
    