import sys
import os

from torch import device, load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resources.rnn import RLRNN
from resources.bandits import AgentSindy, AgentNetwork, AgentQ
from sindy_main import main as sindy_main


def setup_rnn(
    path_model=None, 
    n_actions=2, 
    list_sindy_signals=['xLR', 'xQf', 'xC', 'xCf', 'ca', 'cr', 'cp', 'ca_repeat', 'cQ'], 
    device=device('cpu'),
) -> RLRNN:
    
    state_dict = load(path_model)['model']
    hidden_size = state_dict[list(state_dict.keys())[2]].shape[0]
    rnn = RLRNN(n_actions=n_actions, hidden_size=hidden_size, list_sindy_signals=list_sindy_signals, device=device)
    if path_model is not None:
        rnn.load_state_dict(state_dict)
    return rnn


def setup_agent_rnn(
    path_model=None, 
    n_actions=2, 
    list_sindy_signals=['xLR', 'xQf', 'xC', 'xCf', 'ca', 'cr', 'cp', 'ca_repeat', 'cQ'], 
    device=device('cpu'),
    ) -> AgentNetwork:
    
    rnn = setup_rnn(path_model=path_model, n_actions=n_actions, list_sindy_signals=list_sindy_signals, device=device)
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
        verbose = False,
        hidden_size = hidden_size,
        analysis=False,
    )

    return agent 


def setup_custom_q_agent(
    update_rule: callable = None,
    get_qs: callable = None,
    **kwargs
) -> AgentQ:
    
    class AgentCustom(AgentQ):
        
        def __init__(self, n_actions = 2):
            super().__init__(n_actions, 0, 0)
            
        def update(self, a, r):
            q, c = update_rule(self._q, self._c, a, r)
            self._q = q
            self._c = c
            
        # def get_choice_probs(self):
        #     return get_choice_probs(self._q, self._c)
        
        @property
        def q(self):
            return get_qs(self._q, self._c)
        
    return AgentCustom()
    
    
    


if __name__ == '__main__':
    
    setup_agent_sindy(
        model = 'params/benchmarking/sugawara2021_143_4.pkl',
        data = 'data/sugawara2021_143_processed.csv',
    )