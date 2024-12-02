import sys
import os

from torch import device, load
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn import RLRNN
from resources.bandits import AgentSindy, AgentNetwork, AgentQ
from sindy_main import main as sindy_main


def setup_rnn(
    path_model=None, 
    n_actions=2, 
    hidden_size=8, 
    list_sindy_signals=['xLR', 'xQf', 'xC', 'xCf', 'ca', 'cr', 'cp', 'ca_repeat', 'cQ'], 
    device=device('cpu'),
) -> RLRNN:
    
    rnn = RLRNN(n_actions=n_actions, hidden_size=hidden_size, list_sindy_signals=list_sindy_signals, device=device)
    if path_model is not None:
        rnn.load_state_dict(load(path_model)['model'])
    
    return rnn


def setup_agent_rnn(
    path_model=None, 
    n_actions=2, 
    hidden_size=8, 
    list_sindy_signals=['xLR', 'xQf', 'xC', 'xCf', 'ca', 'cr', 'cp', 'ca_repeat', 'cQ'], 
    device=device('cpu'),
    ) -> AgentNetwork:
    
    rnn = setup_rnn(path_model=path_model, n_actions=n_actions, hidden_size=hidden_size, list_sindy_signals=list_sindy_signals, device=device)
    agent = AgentNetwork(model=rnn, n_actions=n_actions, deterministic=True)
    
    return agent


def setup_agent_sindy(
    model,
    data,
    threshold = 0.03,
    hidden_size = 8,
) -> AgentSindy:
    
    agent, _, _ = sindy_main(
        model = model,
        data = data,
        threshold = threshold,
        verbose = True,
        hidden_size = hidden_size,
        analysis=False,
    )

    return agent 


def setup_benchmark_q_agent(
    parameters,
    **kwargs
) -> AgentQ:
    
    class AgentBenchmark(AgentQ):
        
        def __init__(self, parameters, n_actions = 2):
            super().__init__(n_actions, 0, 0)
            
            self._parameters = parameters
            
        def update(self, a, r):
            # q, c = update_rule(self._q, self._c, a, r)
            ch = np.eye(2)[a]

            # Compute prediction errors for each outcome
            rpe = (r - self._q) * ch
            cpe = ch - self._c
            
            # Update values
            lr = np.where(r > 0.5, self._parameters['alpha_pos'], self._parameters['alpha_neg'])
            self._q += lr * rpe
            self._c += self._parameters['alpha_c'] * cpe
            
        @property
        def q(self):
            return self._parameters['beta_r'] * self._q + self._parameters['beta_c'] * self._c
        
    return AgentBenchmark(parameters)
    
    
    


if __name__ == '__main__':
    
    setup_agent_sindy(
        model = 'params/benchmarking/sugawara2021_143_4.pkl',
        data = 'data/sugawara2021_143_processed.csv',
    )