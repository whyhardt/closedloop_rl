import torch

from theorist import rl_sindy_theorist
from utils.convert_dataset import convert_dataset


path_data = 'data/2arm/sugawara2021_143_processed.csv'
dataset = convert_dataset(path_data)[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rl_sindy = rl_sindy_theorist(
    n_participants=dataset.xs.shape[0],
    device=device,
    verbose=True,
    epochs=16,
    )

rl_sindy.fit(dataset.xs.numpy(), dataset.ys.numpy())
rl_sindy.predict(dataset.xs.numpy())
