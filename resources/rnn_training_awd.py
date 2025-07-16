import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

from copy import deepcopy # For copying the wholemodel
import time
import numpy as np
from resources.rnn import BaseRNN, CustomEmbedding
from resources.rnn_utils import DatasetRNN

# Create Mask TODO: Find out what this does
def apply_mask(preds, ss):
    mask = ss[..., :1] > -1
    return preds * mask

def fit_with_metaopt(
    model: BaseRNN,
    dataset_train: DatasetRNN,
    dataset_val: DatasetRNN = None,
    model_optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-7,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    scheduler: bool = False,
    n_steps: int = -1,
    verbose: bool = True,
    path_save_checkpoints: str = None,
    ):

    """
    """
    # Set batch size
    if batch_size == -1:
        batch_size = len(dataset_train)

    # Set bagging
    if bagging:
        batch_size = max(batch_size, 64)  # Ensure batch size is at least 64 for bagging
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size)
    else:
        raise NotImplementedError("Bagging=False is not implemented in this function.")
    
    # Init DataLoaders
    # num_cores = 4 
    num_workers = 0 # 0 on PC 2 on laptop
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError("Validation dataset is required for training with meta-optimization.")


    # Set up learning rate scheduler
    warmup_steps = 1024 if epochs > 1024 else 1 #int(epochs * 0.125/16)
    if scheduler and model_optimizer is not None:
        default_lr = model_optimizer.param_groups[0]['lr'] + 0

        def warmup_lr_lambda(current_step):
            scale = 1e-1 / default_lr
            if current_step < warmup_steps * 0.8:
                return 0.1 * scale  # Scaling factor during the first 80% of warmup
            elif current_step < warmup_steps:
                # Linearly anneal towards 1.0 in the last 20% of warmup steps
                progress = (current_step - warmup_steps * 0.8) / (warmup_steps * 0.2)
                return (0.1 + progress * (1.0 - 0.1)) * scale
            else:
                return 1.0  # Default learning rate scaling after warmup
        
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda=warmup_lr_lambda)
                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=model_optimizer, T_0=warmup_steps if warmup_steps > 0 else 64, T_mult=2)
    else:
        scheduler_warmup, scheduler = None, None

    # Epoch error handling
    if epochs == 0:
        raise ValueError("Epochs must be specified.")

    loss_fn = nn.CrossEntropyLoss()
    
    # Define some maybe useless variables
    convergence_value = 1
    last_loss = 1
    recency_factor = 0.5
    train_loss = torch.tensor(0.)
    val_loss = torch.tensor(0.) # 0 for the first 10 epochs
    save_at_epoch = warmup_steps

    # Initialize hypergrad if we use update freq
    hypergrad = torch.tensor(0.)

    # Histories for plotting
    history_train_loss = []
    history_val_loss = []
    history_log_lambda = []
    history_hypergrad = []

    # Set up meta-optimization
    # Now using normal lambda
    prev_lambda = torch.tensor(0.0) # Init at 0
    lambda_val = 0.0  # Initial value for lambda, make sure its float
    ema_scale = 0.1  # Exponential moving average scale: lower = more smoothing

    awd_scale = 0.1 # 0.1 is scaling from AWD paper (Ghiasi et al., 2023) # This is basically a new hyperparameter


    # Training loop
    try:
        for epoch in range(epochs):
            # Meaasure the time
            t_start = time.time()

            # Set up the data:
            # Training set
            xs,ys = next(iter(dataloader_train))
            xs, ys = xs.to(model.device), ys.to(model.device)
            # Validation set
            xs_val, ys_val = next(iter(dataloader_val))
            xs_val, ys_val = xs_val.to(model.device), ys_val.to(model.device)

            model.set_initial_state(batch_size=len(xs))

            # Normal training step
            model_optimizer.zero_grad()
            state = model.get_state(detach=True)
            model.train()
            outputs = model(xs, state, batch_first=True)[0]
            params = torch.cat([p.view(-1) for p in model.parameters()])
            l2_norm = (params ** 2).mean()
            outputs = apply_mask(outputs, xs)
            # Calculate the training loss without regularization
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                 torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                 )
            
            # Backprop step to get the gradients
            train_loss.backward(retain_graph=True)

            # AWD meta-optimization step TODO: Find out how this works
            # 1. Compute average gradient and weight norms
            grad_norm = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))
            weight_norm = torch.sqrt(sum((p.data ** 2).sum() for p in model.parameters()))
            current_lambda = (awd_scale * grad_norm) / (weight_norm + 1e-8)  # Avoid dividing by zero

            # 2. Apply EMA smoothing
            lambda_val = ema_scale * current_lambda + (1 - ema_scale) * prev_lambda.detach()
            prev_lambda = lambda_val
            lambda_val = current_lambda

            # 3. Update train loss with regularization
            train_loss += lambda_val * l2_norm

            # Update only after AWD
            # May be able to backward only once TODO: Implement this
            model_optimizer.zero_grad()
            train_loss.backward()
            #
            model_optimizer.step()

            # Validation step
            model.eval()
            state = model.get_state(detach=True)
            val_preds = model(xs_val, state, batch_first=True)[0]
            val_preds = apply_mask(val_preds, xs_val)

            val_loss = loss_fn(
                val_preds.reshape(-1, model._n_actions),
                torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1), )

            # Put lambda in log scale for logging
            log_lambda = torch.log(torch.tensor(lambda_val))

            # Save histories
            history_train_loss.append(train_loss.item())
            history_val_loss.append(val_loss.item())
            history_log_lambda.append(log_lambda.item())
            history_hypergrad.append(hypergrad.item())

            # Check for convergence
            # Note taht val_loss here is calculated in the inner loop
            dloss = last_loss - val_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = val_loss 

            # Endo of epoch message
            msg = None
            if verbose:
                # Shortened num decimal points displayed
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(Val): {val_loss:.4f}"
                msg += f"; log_lambda: {log_lambda.item():.4f}"
                msg += f"; hypergrad: {hypergrad.item():.7f}"
                msg += f"; Time: {time.time()-t_start:.2f}"

                # Just to use convergence_value
                if converged:
                    msg += " --- Converged!"

                print(msg)
                # Missing convergence back here - not necessary for my metaopt
                # Missing scheduler lr info here - not necessary for my metaopt

            # Scheduler update
            # Only with CosineAnnealingWarmRestarts
            if scheduler is not None:
                if epoch < warmup_steps:
                    scheduler_warmup.step()
                else:
                    scheduler.step()
                
            # Save checkpoint TODO: Test if this still works on correct epochs
            if path_save_checkpoints and (epoch + 1) == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace(".", f"_ep{epoch + 1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        msg = "Training interrupted. Continuing with further operations..."
        
        # Wrap histories 
        histories = [history_train_loss, history_val_loss, history_log_lambda, history_hypergrad]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_val_loss, history_log_lambda, history_hypergrad]
    return model, model_optimizer, histories
