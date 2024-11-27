from torch.nn import BatchNorm1d
import torch

t = torch.randn((1024, 4, 2))

bn = BatchNorm1d(4)
bn.train()

epochs = 1024
batch_size = 1024
for i in range(epochs):
    t_batch = t[torch.randint(0, len(t), (batch_size,))]
    bn(t_batch)

print('BN in training mode:')
print(bn(t[:1]))

bn.eval()
print('BN in eval mode:')
print(bn(t[:1]))

print('Running stats:')
print(f'Mean = {bn.running_mean}')
print(f'Std = {bn.running_var}')
