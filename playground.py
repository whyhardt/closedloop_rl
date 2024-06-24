import pysindy as ps
import numpy as np

# library_setup = {
#     'xQf': [],
#     'xQr': ['cr'],
#     'xH': ['ca[k-1]']
# }

# libraries = {key: None for key in library_setup.keys()}
# feature_names = {key: [key] + library_setup[key] for key in library_setup.keys()}
# for key in library_setup.keys():
#     library = ps.PolynomialLibrary(degree=2)
#     library.fit(np.random.rand(10, len(feature_names[key])))
#     # print(library.get_feature_names_out(feature_names[key]))
#     libraries[key] = library

# feature_names = ['xQf', 'xQr', 'xH', 'ca', 'ca[k-1]', 'cr']
# library = ps.TensoredLibrary([ps.PolynomialLibrary(degree=2) for _ in range(3)],
#                              inputs_per_library=np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1], [0, 0, 1, 0, 1, 0]]),
#                              )

# library.fit(np.random.rand(10, len(feature_names)))
# print(library.get_feature_names(feature_names))