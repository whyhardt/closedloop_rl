"""Utility functions for training RNNs."""
from __future__ import print_function

from typing import Any, Callable, Dict, Optional, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time
import warnings

warnings.filterwarnings("ignore")



def find_session_end(x):
  """Get last trial of session."""
  # If the last 2 entries are -1, assume the session has been padded + find true session end.
  if np.all(x[-2:] < 0):
    trial_end = np.where((x[:-1] > 0) & (np.diff(x != 0)))[0][-1]
  else:
    trial_end = x.shape[0]
  return trial_end


class DatasetRNN:
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
      msg = ('number of episodes in xs {} must be equal to number of episodes'
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

  def __next__(self) -> Tuple[chex.Array, chex.Array]:
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


def nan_in_dict(d):
  """Check a nested dict (e.g. hk.params) for nans."""
  if not isinstance(d, dict):
    return np.any(np.isnan(d))
  else:
    return any(nan_in_dict(v) for v in d.values())


def train_model(
    model_fun: Callable[[], hk.RNNCore],
    dataset: DatasetRNN,
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    random_key: Optional[chex.PRNGKey] = None,
    opt_state: Optional[optax.OptState] = None,
    params: Optional[hk.Params] = None,
    n_steps: int = 1000,
    penalty_scale: float = 0,
    loss_fun: str = 'categorical',
    do_plot: bool = True,
    truncate_seq_length: Optional[int] = None,
    ) -> Tuple[hk.Params, optax.OptState, Dict[str, np.ndarray]]:
  """Trains a model for a fixed number of steps.

  Args:
    model_fun: A function that, when called, returns a Haiku RNN object
    dataset: A DatasetRNN, containing the data you wish to train on
    optimizer: The optimizer you'd like to use to train the network
    random_key: A jax random key, to be used in initializing the network
    opt_state: An optimzier state suitable for opt.
      If not specified, will initialize a new optimizer from scratch.
    params:  A set of parameters suitable for the network given by make_network.
      If not specified, will begin training a network from scratch.
    n_steps: An integer giving the number of steps you'd like to train for
      (default=1000)
    penalty_scale: scalar weight applied to bottleneck penalty (default = 0)
    loss_fun: string specifying type of loss function (default='categorical')
    do_plot: Boolean that controls whether a learning curve is plotted
      (default=True)
    truncate_seq_length: truncate to sequence length (default=None)

  Returns:
    params: Trained parameters
    opt_state: Optimizer state at the end of training
    losses: Losses on both datasets
  """
  n_steps = int(n_steps)
  sample_xs, _ = next(dataset)  # Get a sample input, for shape

  # Haiku, step one: Define the batched network
  def unroll_network(xs):
    core = model_fun()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)
    ys, _ = hk.dynamic_unroll(core, xs, state)
    return ys

  # Haiku, step two: Transform the network into a pair of functions
  # (model.init and model.apply)
  model = hk.transform(unroll_network)

  # PARSE INPUTS
  if random_key is None:
    random_key = jax.random.PRNGKey(0)
  # If params have not been supplied, start training from scratch
  if params is None:
    random_key, key1 = jax.random.split(random_key)
    params = model.init(key1, sample_xs)
  # It an optimizer state has not been supplied, start optimizer from scratch
  if opt_state is None:
    opt_state = optimizer.init(params)

  def categorical_log_likelihood(
      labels: np.ndarray, output_logits: np.ndarray
  ) -> float:
    # Mask any errors for which label is negative
    mask = jnp.logical_not(labels < 0)
    log_probs = jax.nn.log_softmax(output_logits)
    log_liks = labels * log_probs
    masked_log_liks = jnp.multiply(log_liks, mask)
    loss = -jnp.nansum(masked_log_liks)/jnp.prod(jnp.array(masked_log_liks.shape[:-1]))
    return loss

  def categorical_loss(
      params, xs: np.ndarray, labels: np.ndarray, random_key
  ) -> float:
    output_logits = model.apply(params, random_key, xs)
    # jax.debug.breakpoint()
    # jax.debug.print("{x}", x=output_logits)
    loss = categorical_log_likelihood(labels, output_logits)
    return loss

  def penalized_categorical_loss(
      params, xs, targets, random_key, penalty_scale=penalty_scale
    ) -> float:
      """Treats the last element of the model outputs as a penalty."""
      # (n_steps, n_episodes, n_targets)
      model_output = model.apply(params, random_key, xs)
      output_logits = model_output[:, :, :-1]
      penalty = jnp.sum(model_output[:, :, -1])  # ()
      loss = (
          categorical_log_likelihood(targets, output_logits)
          + penalty_scale * penalty
      )
      return loss

  losses = {
      'categorical': categorical_loss,
      'penalized_categorical': penalized_categorical_loss,
  }
  
  compute_loss = jax.jit(losses[loss_fun])
  # compute_loss = losses[loss_fun]

  # Define what it means to train a single step
  @jax.jit
  def train_step(
      params, opt_state, xs, ys, random_key
  ) -> Tuple[float, Any, Any]:
    loss, grads = jax.value_and_grad(compute_loss, argnums=0)(
        params, xs, ys, random_key
    )
    grads, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, grads)
    return loss, params, opt_state

  # Train the network!
  training_loss = []
  t_start = time.time()
  for step in jnp.arange(n_steps):
    random_key, key_i = jax.random.split(random_key, 2)
    # Train on training data
    xs, ys = next(dataset)
    if (truncate_seq_length is not None) and (truncate_seq_length < xs.shape[0]):
      xs = xs[:truncate_seq_length]
      ys = ys[:truncate_seq_length]

    loss, params, opt_state = train_step(params, opt_state, xs, ys, key_i)

    # Log every 10th step
    if step % 10 == 9:
      training_loss.append(float(loss))
      print(f'\rStep {step + 1} of {n_steps}; '
            f'Loss: {loss:.7f}; '
            f'Time: {time.time()-t_start:.1f}s)', end='')

  # If we actually did any training, print final loss and make a nice plot
  if n_steps > 1 and do_plot:
    plt.figure()
    plt.semilogy(training_loss, color='black')
    plt.xlabel('Training Step')
    plt.ylabel('Mean Loss')
    plt.title('Loss over Training')

  losses = {
      'training_loss': np.array(training_loss),
  }

  # Check if anything has become NaN that should not be NaN
  if nan_in_dict(params):
    print(params)
    raise ValueError('NaN in params')
  if len(training_loss) > 0 and np.isnan(training_loss[-1]):
    raise ValueError('NaN in loss')

  return params, opt_state, losses


def fit_model(
    model_fun,
    dataset,
    optimizer = None,
    model_params = None,
    optimizer_state = None,
    loss_fun: str = 'categorical',
    convergence_thresh: float = 1e-5,
    random_key = None,
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
  if random_key is None:
    random_key = jax.random.PRNGKey(0)
  
  if model_params is not None and optimizer_state is not None:
    params = model_params
    opt_state = optimizer_state
  else:
    # Initialize the model
    params, opt_state, _ = train_model(
        model_fun,
        dataset,
        optimizer=optimizer,
        n_steps=0,
    )

  # Train until the loss stops going down
  continue_training = True
  converged = False
  loss = np.inf
  n_calls_to_train_model = 0
  while continue_training:
    params, opt_state, losses = train_model(
        model_fun,
        dataset,
        params=params,
        opt_state=opt_state,
        optimizer=optimizer,
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


def eval_model(
    model_fun: Callable[[], hk.RNNCore],
    params: hk.Params,
    xs: np.ndarray,
) ->  Tuple[np.ndarray, Any]:
  """Run an RNN with specified params and inputs. Track internal state.

  Args:
    model_fun: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    xs: A batch of inputs [timesteps, episodes, features] suitable for the model

  Returns:
    y_hats: Network outputs at each timestep
    states: Network states at each timestep
  """

  n_steps = jnp.shape(xs)[0]

  def unroll_network(xs):
    core = model_fun()
    batch_size = jnp.shape(xs)[1]
    state = core.initial_state(batch_size)

    y_hats = []
    states = []

    for t in range(n_steps):
      states.append(state)
      y_hat, new_state = core(xs[t, :], state)
      state = new_state

      y_hats.append(y_hat)

    return np.asarray(y_hats), states

  model = hk.transform(unroll_network)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  apply = model.apply
  y_hats, states = apply(params, key, xs)

  return np.asarray(y_hats), states


def step_network(
    make_network: Callable[[], hk.RNNCore],
    params: hk.Params,
    state: Any,
    xs: np.ndarray,
) -> Tuple[np.ndarray, Any]:
  """Run an RNN for just a single step on a single input (no batching).

  Args:
    make_network: A Haiku function that defines a network architecture
    params: A set of params suitable for that network
    state: An RNN state suitable for that network
    xs: An input for a single timestep from a single episode, with
      shape [n_features]

  Returns:
    y_hat: The output given by the network, with dimensionality [n_features]
    new_state: The new RNN state of the network
  """

  def step_sub(xs):
    core = make_network()
    y_hat, new_state = core(xs, state)
    return y_hat, new_state

  model = hk.transform(step_sub)
  key = jax.random.PRNGKey(np.random.randint(2**32))
  y_hat, new_state = model.apply(params, key, np.expand_dims(xs, axis=0))

  return y_hat, new_state


def get_initial_state(make_network: Callable[[], hk.RNNCore],
                      params: Optional[Any] = None) -> Any:
  """Get the default initial state for a network architecture.

  Args:
    make_network: A Haiku function that defines a network architecture
    params: Optional parameters for the Hk function. If not passed, will init
      new parameters. For many models this will not affect initial state

  Returns:
    initial_state: An initial state from that network
  """

  # The logic below needs a jax randomy key and a sample input in order to work.
  # But neither of these will affect the initial network state, so its ok to
  # generate throwaways
  random_key = jax.random.PRNGKey(np.random.randint(2**32))

  def unroll_network():
    core = make_network()
    state = core.initial_state(batch_size=1)

    return state

  model = hk.transform(unroll_network)

  if params is None:
    params = model.init(random_key)

  initial_state = model.apply(params, random_key)

  return initial_state

