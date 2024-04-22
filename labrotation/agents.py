import haiku as hk
import numpy as np
from typing import Callable, Iterable, Union

from resources import bandits


class AgentQuadQ(bandits.AgentQ):
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 3.0,
        n_actions: int = 2,
        forgetting_rate: float = 0.0,
        perseveration_bias: float = 0.0,
    ):
        super().__init__(alpha, beta, n_actions, forgetting_rate, perseveration_bias)

    def update(self, choice: int, reward: float):
        """Update the agent after one step of the task.

        Args:
          choice: The choice made by the agent. 0 or 1
          reward: The reward received by the agent. 0 or 1
        """

        # Decay q-values toward the initial value.
        self._q = (
            1 - self._forgetting_rate
        ) * self._q + self._forgetting_rate * self._q_init

        # Update chosen q for chosen action with observed reward.
        self._q[choice] = (
            self._q[choice] - self._alpha * self._q[choice] ** 2 + self._alpha * reward
        )


class AgentSindy(bandits.AgentQ):
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = 3.0,
        n_actions: int = 2,
        forgetting_rate: float = 0.0,
        perservation_bias: float = 0.0,
    ):
        super().__init__(alpha, beta, n_actions, forgetting_rate, perservation_bias)

        self._update_rule = (
            lambda q, choice, reward: (1 - self._alpha) * q[choice]
            + self._alpha * reward
        )
        self._update_rule_formula = None

    def set_update_rule(self, update_rule: callable, update_rule_formula: str = None):
        self._update_rule = update_rule
        self._update_rule_formula = update_rule_formula

    @property
    def update_rule(self):
        if self._update_rule_formula is not None:
            return self._update_rule_formula
        else:
            return f"{self._update_rule}"

    def update(self, choice: int, reward: int):
        for c in range(self._n_actions):
            self._q[c] = self._update_rule(self._q[c], int(c == choice), reward)


class AgentNetwork_VisibleState(bandits.AgentNetwork):
    def __init__(
        self,
        make_network: Callable[[], hk.RNNCore],
        params: hk.Params,
        n_actions: int = 2,
        state_to_numpy: bool = False,
        habit=False,
    ):
        super().__init__(
            make_network=make_network,
            params=params,
            n_actions=n_actions,
            state_to_numpy=state_to_numpy,
        )
        self.habit = habit

    @property
    def q(self):
        if self.habit:
            return self._state[2], self._state[3]
        else:
            return self._state[3].reshape(-1)


dict_agents = {
    "basic": lambda alpha,
    beta,
    n_actions,
    forgetting_rate,
    perseveration_bias: bandits.AgentQ(
        alpha, beta, n_actions, forgetting_rate, perseveration_bias
    ),
    "quad_q": lambda alpha,
    beta,
    n_actions,
    forgetting_rate,
    perseveration_bias: AgentQuadQ(
        alpha, beta, n_actions, forgetting_rate, perseveration_bias
    ),
}


def get_q(
    experiment: bandits.BanditSession,
    agent: Union[bandits.AgentQ, bandits.AgentNetwork, AgentSindy],
):
    """Compute Q-Values of a specific agent for a specific experiment.

    Args:
        experiment (bandits.BanditSession): _description_
        agent (_type_): _description_

    Returns:
        _type_: _description_
    """

    choices = np.expand_dims(experiment.choices, 1)
    rewards = np.expand_dims(experiment.rewards, 1)
    qs = np.zeros((experiment.choices.shape[0], agent._n_actions))
    choice_probs = np.zeros((experiment.choices.shape[0], agent._n_actions))

    agent.new_sess()

    for trial in range(experiment.choices.shape[0]):
        qs[trial] = agent.q
        choice_probs[trial] = agent.get_choice_probs()
        agent.update(int(choices[trial]), float(rewards[trial]))

    return qs, choice_probs


def parse_equation_for_sympy(eq):
    # replace all blank spaces with '*' where necessary
    # only between number and letter in exactly this order
    blanks = [i for i, ltr in enumerate(eq) if ltr == " "]
    for blank in blanks:
        if (eq[blank + 1].isalpha() or eq[blank - 1].isdigit()) and (
            eq[blank + 1].isalpha() or eq[blank + 1].isdigit()
        ):
            eq = eq[:blank] + "*" + eq[blank + 1 :]

    # replace all '^' with '**'
    eq = eq.replace("^", "**")

    # remove all [k]
    eq = eq.replace("[k]", "")

    return eq


def make_sindy_data(
    dataset,
    agent: bandits.AgentQ,
    sessions=-1,
    get_choices=True,
    verbose=False,
    # keep_sessions=False,
):
    # Get training data for SINDy
    # put all relevant signals in x_train

    if not isinstance(sessions, Iterable) and sessions == -1:
        # use all sessions
        sessions = np.arange(len(dataset))
    else:
        # use only the specified sessions
        sessions = np.array(sessions)

    if get_choices:
        n_control = 2
    else:
        n_control = 1

    choices = np.stack([dataset[i].choices for i in sessions], axis=0)
    rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
    qs = np.stack([dataset[i].q for i in sessions], axis=0)

    if not get_choices:
        raise NotImplementedError("Only get_choices=True is implemented right now.")
    else:
        choices_oh = np.zeros((len(sessions), choices.shape[1], agent._n_actions))
        for sess in sessions:
            # one-hot encode choices
            choices_oh[sess] = np.eye(agent._n_actions)[choices[sess]]
            # concatenate all qs values of one sessions along the trial dimension
            qs_all = np.concatenate(
                [
                    np.stack(
                        [
                            np.expand_dims(qs_sess[:, i], axis=-1)
                            for i in range(agent._n_actions)
                        ],
                        axis=0,
                    )
                    for qs_sess in qs
                ],
                axis=0,
            )
            c_all = np.concatenate(
                [
                    np.stack([c_sess[:, i] for i in range(agent._n_actions)], axis=0)
                    for c_sess in choices_oh
                ],
                axis=0,
            )
            r_all = np.concatenate(
                [
                    np.stack([r_sess for _ in range(agent._n_actions)], axis=0)
                    for r_sess in rewards
                ],
                axis=0,
            )

    # get observed dynamics
    x_train = qs_all
    feature_names = ["q"]

    # get control
    control_names = []
    control = np.zeros((*x_train.shape[:-1], n_control))
    if get_choices:
        control[:, :, 0] = c_all
        control_names += ["c"]
    control[:, :, n_control - 1] = r_all
    control_names += ["r"]

    feature_names += control_names

    if verbose:
        print(f"Shape of Q-Values is: {x_train.shape}")
        print(f"Shape of control parameters is: {control.shape}")
        print(f"Feature names are: {feature_names}")

    # make x_train and control sequences instead of arrays
    x_train = [x_train_sess for x_train_sess in x_train]
    control = [control_sess for control_sess in control]

    return x_train, control, feature_names
