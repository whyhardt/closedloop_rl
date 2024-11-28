from torch.nn import BatchNorm1d
import torch

t = torch.arange(0, 24).view(4, 3, 2)
print(t)

print(t.view(12, 2))