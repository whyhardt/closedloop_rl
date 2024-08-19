import pysindy as ps
import numpy as np

data = np.random.random((100,1))

model = ps.SINDy(
    ps.optimizers.STLSQ(),
    ps.PolynomialLibrary(),
    discrete_time=True,
)

model.fit(data)
model.print()

print(model.coefficients())

model.model.steps[-1][1].coef_[0, 0] = 1.

print(model.coefficients())