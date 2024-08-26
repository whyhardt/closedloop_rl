import pysindy as ps
import numpy as np
import matplotlib.pyplot as plt 

q_init = 0.5
reward = np.linspace(0, 1, 100)
q = np.linspace(0, 1, 100)

cb = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        cb[i, j] = (q[i]-q_init)*(reward[j] - 0.5)/2

q_mesh, reward_mesh = np.meshgrid(q, reward)
cb = (q_mesh-q_init)*(reward_mesh - 0.5)/2
plt.imshow(cb, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title('Confirmation bias')
plt.xlabel('Q-Value')
plt.ylabel('Reward')
plt.show()

# plt.plot(cb[0])
# plt.show()