import pysindy as ps
import numpy as np

init_values = [np.random.rand(1).reshape(1, 1) for _ in range(100)]
array = [np.concatenate([init_values[i], init_values[i]+0.35]) for i in range(100)]

model = ps.SINDy(
    ps.optimizers.STLSQ(threshold=0.03),
    ps.feature_library.PolynomialLibrary(degree=2),
    discrete_time=True,
)

model.fit(array, t=1, multiple_trajectories=True)
model.set_params()
print(array[0])
print(model.predict(array[0][-1]))
