import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from typing import Optional, Tuple, Dict
import time


class DatasetRNN(Dataset):
  """Holds a dataset for training an RNN, consisting of inputs and targets.

     Both inputs and targets are stored as [timestep, episode, feature]
     Serves them up in batches
  """

  def __init__(self,
               xs: np.ndarray,
               ys: np.ndarray,
               batch_size: Optional[int] = None):
    """Do error checking and bin up the dataset into batches.

    Args:
      xs: Values to become inputs to the network.
        Should have dimensionality [timestep, episode, feature]
      ys: Values to become output targets for the RNN.
        Should have dimensionality [timestep, episode, feature]
      batch_size: The size of the batch (number of episodes) to serve up each
        time next() is called. If not specified, all episodes in the dataset 
        will be served
    """

    if batch_size is None:
      batch_size = xs.shape[1]

    # Error checking
    # Do xs and ys have the same number of timesteps?
    if xs.shape[0] != ys.shape[0]:
      msg = ('number of timesteps in xs {} must be equal to number of timesteps'
             ' in ys {}.')
      raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

    # Do xs and ys have the same number of episodes?
    if xs.shape[1] != ys.shape[1]:
      msg = ('number of timesteps in xs {} must be equal to number of timesteps'
             ' in ys {}.')
      raise ValueError(msg.format(xs.shape[0], ys.shape[0]))

    # Is the number of episodes divisible by the batch size?
    if xs.shape[1] % batch_size != 0:
      msg = 'dataset size {} must be divisible by batch_size {}.'
      raise ValueError(msg.format(xs.shape[1], batch_size))

    # Property setting
    self._xs = xs
    self._ys = ys
    self._batch_size = batch_size
    self._dataset_size = self._xs.shape[1]
    self._idx = 0
    self.n_batches = self._dataset_size // self._batch_size

  def __iter__(self):
    return self

  def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
    """Return a batch of data, including both xs and ys.
    
    Returns:
      x, y: next input (x) and target (y) in sequence.
    """

    # Define the chunk we want: from idx to idx + batch_size
    start = self._idx
    end = start + self._batch_size
    # Check that we're not trying to overshoot the size of the dataset
    assert end <= self._dataset_size

    # Update the index for next time
    if end == self._dataset_size:
      self._idx = 0
    else:
      self._idx = end

    # Get the chunks of data
    x, y = self._xs[:, start:end], self._ys[:, start:end]

    return x, y

def train_model(
    model: nn.Module,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer = None,
    random_seed: Optional[int] = None,
    n_steps: int = 1000,
    penalty_scale: float = 0,
    loss_fun: str = 'categorical',
    do_plot: bool = True,
    truncate_seq_length: Optional[int] = None,
) -> Tuple[torch.nn.Parameter, Dict[str, np.ndarray]]:
    """Trains a model for a fixed number of steps.

    Args:
        model: A torch Module representing the model you want to train
        dataset: A DatasetRNN, containing the data you wish to train on
        optimizer: The optimizer you'd like to use to train the network
        random_seed: A random seed to ensure reproducibility
        n_steps: An integer giving the number of steps you'd like to train for (default=1000)
        penalty_scale: scalar weight applied to bottleneck penalty (default = 0)
        loss_fun: string specifying type of loss function (default='categorical')
        do_plot: Boolean that controls whether a learning curve is plotted (default=True)
        truncate_seq_length: truncate to sequence length (default=None)

    Returns:
        model: Trained model parameters
        losses: Losses on both datasets
    """
    torch.manual_seed(random_seed) if random_seed else None
    sample_xs, _ = next(dataset)  # Get a sample input, for shape

    if optimizer is None:
      optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    def categorical_log_likelihood(labels: np.ndarray, output_logits: np.ndarray) -> float:
        mask = labels >= 0
        log_probs = nn.functional.log_softmax(torch.tensor(output_logits), dim=-1)
        log_liks = labels * log_probs.numpy()
        masked_log_liks = np.multiply(log_liks, mask)
        loss = -np.nansum(masked_log_liks) / np.prod(np.array(masked_log_liks.shape[:-1]))
        return loss

    def categorical_loss(
        model, xs: np.ndarray, labels: np.ndarray
    ) -> float:
        output_logits = model(torch.tensor(xs))
        loss = categorical_log_likelihood(labels, output_logits.detach().numpy())
        return loss

    def penalized_categorical_loss(
        model, xs, targets, penalty_scale=penalty_scale
    ) -> float:
        """Treats the last element of the model outputs as a penalty."""
        model_output = model(torch.tensor(xs))
        output_logits = model_output[:, :, :-1]
        penalty = np.sum(model_output[:, :, -1])  # ()
        loss = (
            categorical_log_likelihood(targets, output_logits.detach().numpy())
            + penalty_scale * penalty
        )
        return loss

    losses = {
        'categorical': categorical_loss,
        'penalized_categorical': penalized_categorical_loss,
    }

    compute_loss = losses[loss_fun]

    training_loss = []
    for step in range(n_steps):
        xs, ys = next(dataset)
        if truncate_seq_length is not None and truncate_seq_length < xs.shape[0]:
            xs = xs[:truncate_seq_length]
            ys = ys[:truncate_seq_length]

        optimizer.zero_grad()
        loss = compute_loss(model, xs, ys)
        loss.backward()
        optimizer.step()

        training_loss.append(float(loss))

        if step % 10 == 9:
            print(f'\rStep {step + 1} of {n_steps}; Loss: {loss:.7f}', end='')

    if n_steps > 1 and do_plot:
        plt.figure()
        plt.semilogy(training_loss, color='black')
        plt.xlabel('Training Step')
        plt.ylabel('Mean Loss')
        plt.title('Loss over Training')

    losses = {'training_loss': np.array(training_loss)}

    return model, losses
  
def fit_model(
    model: nn.Module,
    dataset: Dataset,
    optimizer: None,
    random_seed: Optional[int] = None,
    loss_fun: str = 'categorical',
    convergence_thresh: float = 1e-5,
    n_steps_per_call: int = 500,
    n_steps_max: int = 2000,
    batch_size: int = 128,
    ):
  """Fits a model to convergence, by repeatedly calling train_model.
  
  Args:
    model_fun: A function that, when called, returns a Haiku RNN object
    dataset: A DatasetRNN, containing the data you wish to train on
    optimizer: The optimizer you'd like to use to train the network
    loss_fun: string specifying type of loss function (default='categorical')
    convergence_thresh: float, the fractional change in loss in one timestep must be below
      this for training to end (default=1e-5).
    random_key: A jax random key, to be used in initializing the network
    n_steps_per_call: The number of steps to give to train_model (default=1000)
    n_steps_max: The maximum number of iterations to run, even if convergence
      is not reached (default=1000)
  """
  torch.manual_seed(random_seed) if random_seed else None
  
  if optimizer is None:
    optimizer = torch.optim.Optimizer = optim.Adam(model.parameters(), lr=1e-3),
  
  # Train until the loss stops going down
  continue_training = True
  converged = False
  loss = np.inf
  n_calls_to_train_model = 0
  while continue_training:
    params, opt_state, losses = train_model(
        model,
        dataset,
        optimizer=optimizer,
        random_seed=random_seed,
        loss_fun=loss_fun,
        do_plot=False,
        n_steps=n_steps_per_call,
    )
    n_calls_to_train_model += 1
    t_start = time.time()

    loss_new = losses['training_loss'][-1]
    # Declare "converged" if loss has not improved very much (but has improved)
    if not np.isinf(loss): 
      convergence_value = np.abs((loss_new - loss)) / loss
      converged = convergence_value < convergence_thresh

    # Check if converged + print status.
    if converged:
      msg = '\nModel Converged!'
      continue_training = False
    elif (n_steps_per_call * n_calls_to_train_model) >= n_steps_max:
      msg = '\nMaximum iterations reached'
      if np.isinf(loss):
        msg += '.'
      else:
        msg += f', but model has reached \nconvergence value of {convergence_value:0.7g} which is greater than {convergence_thresh}.'
      continue_training = False
    else:
      update_msg = '' if np.isinf(loss) else f'(convergence_value = {convergence_value:0.7g}) '
      msg = f'\nModel not yet converged {update_msg}- Running more steps of gradient descent.'
    print(msg + f' Time elapsed = {time.time()-t_start:0.1}s.')
    loss = loss_new

  return params, opt_state, loss
