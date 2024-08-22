import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt 

sigmoid = lambda x: 1/(1+np.exp(-x))

weights = [1, 2, 4, 8, 16]
data = np.linspace(-5, 5)

for w in weights:
    y = np.exp(w*data)

    plt.plot(data, y)
plt.ylim([0,100])
plt.show()


import torch

torch.nn.functional.relu