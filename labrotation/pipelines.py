# Building blocks for reusable model pipelines
import numpy as np

from typing import Iterable

from sklearn.pipeline import FunctionTransformer

from resources import bandits


def make_sindy_dataset(
    dataset: list[bandits.BanditSession],
    n_actions: int,
    sessions: int | Iterable,
    get_choices: bool,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    if not isinstance(sessions, Iterable) and sessions == -1:
        # use all sessions
        sessions = np.arange(len(dataset))
    else:
        # use only the specified sessions
        sessions = np.array(sessions)

    n_control = 2 if get_choices else 1
    choices = np.stack([dataset[i].choices for i in sessions], axis=0)
    rewards = np.stack([dataset[i].rewards for i in sessions], axis=0)
    qs = np.stack([dataset[i].q for i in sessions], axis=0)

    if not get_choices:
        raise NotImplementedError("Only get_choices=True is implemented right now.")
    choices_oh = np.zeros((len(sessions), choices.shape[1], n_actions))
    for sess in sessions:
        # one-hot encode choices
        choices_oh[sess] = np.eye(n_actions)[choices[sess]]
        # concatenate all qs values of one sessions along the trial dimension
        qs_all = np.concatenate(
            [
                np.stack(
                    [np.expand_dims(qs_sess[:, i], axis=-1) for i in range(n_actions)],
                    axis=0,
                )
                for qs_sess in qs
            ],
            axis=0,
        )
        c_all = np.concatenate(
            [
                np.stack([c_sess[:, i] for i in range(n_actions)], axis=0)
                for c_sess in choices_oh
            ],
            axis=0,
        )
        r_all = np.concatenate(
            [
                np.stack([r_sess for _ in range(n_actions)], axis=0)
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

    # TODO: Export this into logging
    # print(f'Shape of Q-Values is: {x_train.shape}')
    # print(f'Shape of control parameters is: {control.shape}')
    # print(f'Feature names are: {feature_names}')

    # make x_train and control sequences instead of arrays
    x_train = list(x_train)
    control = list(control)

    return x_train, control, feature_names


SINDyDatasetConverter = FunctionTransformer(make_sindy_dataset)
