from typing import Callable, Tuple, Iterable, Union
import haiku as hk

from resources import bandits


class AgentQuadQ(bandits.AgentQ):

    def __init__(
            self,
            alpha: float = 0.2,
            beta: float = 3.,
            n_actions: int = 2,
            forgetting_rate: float = 0.,
            perseveration_bias: float = 0.,
    ):
        super().__init__(alpha, beta, n_actions, forgetting_rate, perseveration_bias)

    def update(self,
               choice: int,
               reward: float):
        """Update the agent after one step of the task.

        Args:
          choice: The choice made by the agent. 0 or 1
          reward: The reward received by the agent. 0 or 1
        """

        # Decay q-values toward the initial value.
        self._q = (1 - self._forgetting_rate) * self._q + self._forgetting_rate * self._q_init

        # Update chosen q for chosen action with observed reward.
        self._q[choice] = self._q[choice] - self._alpha * self._q[choice] ** 2 + self._alpha * reward


class AgentSindy(bandits.AgentQ):

    def __init__(
            self,
            alpha: float = 0.2,
            beta: float = 3.,
            n_actions: int = 2,
            forgetting_rate: float = 0.,
            perservation_bias: float = 0., ):
        super().__init__(alpha, beta, n_actions, forgetting_rate, perservation_bias)

        self._update_rule = lambda q, choice, reward: (1 - self._alpha) * q[choice] + self._alpha * reward
        self._update_rule_formula = None

    def set_update_rule(self, update_rule: callable, update_rule_formula: str = None):
        self._update_rule = update_rule
        self._update_rule_formula = update_rule_formula

    def get_update_rule(self):
        return self._update_rule

    @property
    def update_rule(self):
        if self._update_rule_formula is not None:
            return self._update_rule_formula
        else:
            return f'{self._update_rule}'

    def update(self, choice: int, reward: int):

        for c in range(self._n_actions):
            self._q[c] = self._update_rule(self._q[c], int(c == choice), reward)


class AgentNetwork_VisibleState(bandits.AgentNetwork):

    def __init__(self,
                 make_network: Callable[[], hk.RNNCore],
                 params: hk.Params,
                 n_actions: int = 2,
                 state_to_numpy: bool = False,
                 habit=False):
        super().__init__(make_network=make_network, params=params, n_actions=n_actions, state_to_numpy=state_to_numpy)
        self.habit = habit

    @property
    def q(self):
        if self.habit:
            return self._state[2], self._state[3]
        else:
            return self._state[3].reshape(-1)
