import torch
from typing import OrderedDict

logits1 = torch.tensor([[[(0.5)*3, 0.]]], dtype=float)
logits2 = torch.tensor([[[(.5+.6)*3, .6*3]]], dtype=float)
logits = torch.concat((logits1, logits2), dim=0)
print(logits.shape)
print(logits)

softmax = torch.nn.Softmax(2)

print(softmax(logits))