import numpy as np
from sampling import sample_custom_library


if __name__ == "__main__":
    custom_lib_functions = [
        # sub-library which is always included
        lambda q, c, r: q,
        lambda q, c, r: r,
        lambda q, c, r: np.power(q, 2),
        lambda q, c, r: q * r,
        lambda q, c, r: np.power(r, 2),
        # sub-library if the possible action was chosen
        lambda q, c, r: c,
        lambda q, c, r: c * q,
        lambda q, c, r: c * r,
        lambda q, c, r: c * np.power(q, 2),
        lambda q, c, r: c * q * r,
        lambda q, c, r: c * np.power(r, 2),
    ]

    custom_lib_names = [
        # sub library which is always included
        "q",
        "r",
        "q^2",
        "q*r",
        "r^2",
        # sub library if the possible action was chosen
        "c",
        "c*q",
        "c*r",
        "c*q^2",
        "c*q*r",
        "c*r^2",
    ]

    sampled_functions, sampled_function_names, weights = sample_custom_library(
        custom_lib_functions, custom_lib_names, np.random.normal
    )

    assert len(sampled_functions) == len(
        sampled_function_names
    ), "The length of the functions and their names should be the same"
    for i in range(len(sampled_functions)):
        print(sampled_functions[i], sampled_function_names[i], weights[i])
