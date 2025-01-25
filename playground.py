import torch

from theorist_basic import rl_sindy_theorist
from utils.convert_dataset import convert_dataset


path_data = 'data/2arm/sugawara2021_143_processed.csv'
dataset = convert_dataset(path_data)[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rl_sindy = rl_sindy_theorist(
    n_participants=dataset.xs[dataset.features_xs[0]].shape[0],
    device=device,
    verbose=True,
    epochs=16,
    )

rl_sindy.fit(dataset.xs, dataset.ys)
rl_sindy.predict(dataset.xs)
