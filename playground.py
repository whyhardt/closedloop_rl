import numpy as np

values = []
for i in range(100):
    value = np.abs(np.random.normal(scale=0.1))
    print(value)
    values.append(value)
print(np.max(values))
    