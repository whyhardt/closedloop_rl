import sys
import os

from torch import device, load
import numpy as np
from typing import List
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn import RLRNN
from resources.bandits import AgentSindy, AgentNetwork, AgentQ
from sindy_main import main as sindy_main


def setup_rnn(
    path_model,
    n_participants, 
    n_actions=2, 
    hidden_size=8,
    participant_emb=False,
    counterfactual=False,
    list_sindy_signals=['x_V_LR', 'x_V_nc', 'x_C', 'x_C_nc', 'c_a', 'c_r', 'c_p', 'c_a_repeat', 'c_V'], 
    device=device('cpu'),
) -> RLRNN:
    
    rnn = RLRNN(
        n_actions=n_actions, 
        hidden_size=hidden_size, 
        n_participants=n_participants if participant_emb else 0, 
        signals=list_sindy_signals, 
        device=device, 
        counterfactual=counterfactual,
        )
    rnn.load_state_dict(load(path_model, map_location=torch.device('cpu'))['model'])
    
    return rnn


def setup_agent_rnn(
    path_model,
    n_participants, 
    n_actions=2, 
    hidden_size=8,
    participant_emb=False,
    counterfactual=False, 
    list_sindy_signals=['x_V_LR', 'x_V_LR_cf', 'x_V_nc', 'x_C', 'x_C_nc', 'c_a', 'c_r', 'c_p', 'c_r_cf', 'c_p_cf', 'c_a_repeat', 'c_V'], 
    device=device('cpu'),
    ) -> AgentNetwork:
    
    rnn = setup_rnn(path_model=path_model, n_participants=n_participants, n_actions=n_actions, hidden_size=hidden_size, list_sindy_signals=list_sindy_signals, device=device, participant_emb=participant_emb, counterfactual=counterfactual)
    agent = AgentNetwork(model=rnn, n_actions=n_actions, deterministic=True)
    
    return agent


def setup_agent_sindy(
    model,
    data,
    threshold = 0.03,
    session_id: int = None,
    polynomial_degree: int = 2,
) -> List[AgentSindy]:
    
    agent, _, _ = sindy_main(
        model = model,
        data = data,
        threshold = threshold,
        polynomial_degree=polynomial_degree,
        verbose = True,
        analysis=False,
        session_id=session_id,
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
            
        def update(self, a, r, *args):
            # q, c = update_rule(self._q, self._c, a, r)
            ch = np.eye(2)[a]
            r = r[0]
            
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