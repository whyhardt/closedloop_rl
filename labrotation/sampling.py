from typing import Callable
import numpy as np
import pysindy as ps

from functools import partial


def sample_custom_library(
    lib_functions: list[Callable],
    lib_names: list[Callable],
    distribution: Callable,
    rng=np.random.default_rng(),
) -> tuple[list[Callable], list[Callable], list[float], np.ndarray]:
    """Subsamples lists of library functions and their corresponding names using a given probability function.

    Parameters
    ----------
    lib_functions : list[callable]
        The library functions
    lib_names : list[callable]
        Names of the library functions
    distribution : callable
        Expects a numpy probability distribution or a partial of that
    rng : np.random.Generator, optional
        _description_, by default np.random.default_rng()

    Returns
    -------
    tuple[list[callable], list[callable], list[float]]
        The subsampled lists and their weights.

    Example
    -------
    Say you want to modify the values of a normal distribution to mean=2

    >>> rng = np.random.default_rng() # Initialize the random generator
    >>> normal_func = partial(rng.normal, loc=2)
    >>> sampled_func, sampled_names, weights = sample_custom_library(custom_lib_functions, custom_lib_names, normal_func)
    """

    if len(lib_functions) != len(lib_names):
        raise ValueError(
            f"Length of library functions {len(lib_functions)} does not match their names {len(lib_names)}!"
        )

    dist_func = partial(distribution, size=len(lib_functions))

    # Get the probabilities to draw from

    probs = dist_func()
    norm_probs = np.abs(probs)
    norm_probs -= np.min(norm_probs) # / (np.max(probs) - np.min(probs))
    norm_probs /= np.max(norm_probs) - np.min(norm_probs)  # added
    # norm_probs /= np.sum(norm_probs)

    # scale probs between -1 and 1
    probs /= np.max(np.abs(probs))

    # choices = set(
    #     rng.choice(np.arange(len(lib_functions)), len(lib_functions), p=norm_probs)
    # )
    
    # get the selected functions and the corresponding names and weights
    choices = norm_probs > 0.5
    sampled_funcs = np.array(lib_functions)[choices].tolist()
    sampled_names = np.array(lib_names)[choices].tolist()
    weights = np.array(probs)[choices].tolist()
    choices = choices.tolist()

    # sampled_funcs = [lib_functions[i] for i in choices]
    # sampled_names = [lib_names[i] for i in choices]

    # choice_mask = np.zeros(len(lib_functions), dtype=bool)
    # choice_mask[list(choices)] = True
    return tuple(sampled_funcs), tuple(sampled_names), tuple(weights), tuple(choices)


def generate_libraries(
    custom_lib_functions, custom_lib_names, num_samples: int, distribution: Callable
) -> tuple[list[ps.CustomLibrary], list[list[float]], np.ndarray]:
    
    assert len(custom_lib_functions) == len(custom_lib_names), "Length of library functions does not match their names!"

    libraries = []
    weights = []
    choices = np.empty((num_samples, len(custom_lib_functions)), dtype=bool)

    for i in range(num_samples):
        sampled_func, sampled_name, sampled_weight, choice_mask = sample_custom_library(
            custom_lib_functions, custom_lib_names, distribution
        )
        libraries.append(
            ps.CustomLibrary(sampled_func, sampled_name, include_bias=True)
        )
        weights.append(sampled_weight)
        choices[i] = choice_mask

    return libraries, weights, choices
