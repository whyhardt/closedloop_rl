import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

import higher # For differentiable optimization
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
    update_freq = 5
    initial_log_lambda = -4.0  # Initial value for log_lambda
    lr_log_lambda = 1.0  # Learning rate for log_lambda
    momentum_log_lambda = 0.0  # Momentum for log_lambda TODO: Try out some
    scale_hypergrad = 10  # Scaling factor for hypergrad TODO: Find literature for this

    # I am also clipping the hypergrads, its in the loop
    log_lambda = torch.tensor(initial_log_lambda, requires_grad=True, device=model.device)
    lambda_optimizer = optim.SGD([log_lambda], lr=lr_log_lambda, momentum=momentum_log_lambda)

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

            # Get real lambda value since we use log_lambda only to optimize 
            lambda_val = torch.exp(log_lambda)

            # Try this
            model.set_initial_state(batch_size=len(xs))

            if (epoch + 1) % update_freq == 0:
                # Start the inner loop for meta-optimization
                with higher.innerloop_ctx(model, model_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                    # Set fmodel to training mode
                    fmodel.train()
                    # Set initial state since we use RNN
                    fmodel.set_initial_state(batch_size=len(xs))
                    
                    # T1 step
                    state = model.get_state(detach=True)
                    train_preds = fmodel(xs, state, batch_first=True)[0]

                    # Add mask to preds
                    train_preds = apply_mask(train_preds, xs)

                    # Calculate L2
                    params = torch.cat([p.view(-1) for p in fmodel.parameters()])
                    l2_norm = (params ** 2).mean()


                    train_loss = loss_fn(
                        train_preds.reshape(-1, model._n_actions),
                        torch.argmax(ys.reshape(-1, model._n_actions), dim=1),) + lambda_val * l2_norm

                    # Step with the inner loop optimizer
                    diffopt.step(train_loss)

                    # T2 step
                    # Predict and compute val loss
                    # Set fmodel to eval mode
                    fmodel.eval()
                    fmodel.set_initial_state(batch_size=len(xs_val))
                    state = fmodel.get_state(detach=True)
                    val_preds = fmodel(xs_val, state, batch_first=True)[0]
                    val_preds = apply_mask(val_preds, xs_val)

                    val_loss = loss_fn(
                        val_preds.reshape(-1, model._n_actions),
                        torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1), )

                    # Recompute train loss for next step (after diffopt.step)
                    fmodel.train()
                    fmodel.set_initial_state(batch_size=len(xs))
                    state = fmodel.get_state(detach=True)
                    train_preds = fmodel(xs, state, batch_first=True)[0]
                    params = torch.cat([p.view(-1) for p in fmodel.parameters()])
                    l2_norm = (params ** 2).mean()
                    train_preds = apply_mask(train_preds, xs)
                    train_loss_updated = loss_fn(train_preds.reshape(-1, model._n_actions), 
                                                torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                                ) + lambda_val * l2_norm

                    # Gradients w.r.t. model params
                    train_grads = torch.autograd.grad(train_loss_updated, fmodel.parameters(), create_graph=True)

                    val_grads = torch.autograd.grad(val_loss, fmodel.parameters(), retain_graph=False)

                    # Dot product #?
                    dot = sum((v * t).sum() for v, t in zip(val_grads, train_grads))
                    # Hypergrad: dot product (?) w.r.t. log_lambda
                    hypergrad = torch.autograd.grad(dot, log_lambda)[0]

                # Clip hypergrad and potentially scale it
                hypergrad = torch.clamp(hypergrad, -5.0, 5.0) * scale_hypergrad

                # Update of log_lambda
                # Set gradient of log_lambda to calculated hypergrad
                log_lambda.grad = hypergrad
                # Pytorch step function
                lambda_optimizer.step()
                # Reset gradients
                lambda_optimizer.zero_grad()

            # Normal training step
            model_optimizer.zero_grad()
            state = model.get_state(detach=True)
            model.train()
            outputs = model(xs, state, batch_first=True)[0]
            params = torch.cat([p.view(-1) for p in model.parameters()])
            l2_norm = (params ** 2).mean()
            outputs = apply_mask(outputs, xs)
            # Detach lambda_val from this computation graph so this update does not influence hypergrad calculation
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                 torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                 ) + lambda_val.detach() * l2_norm

            # Backprop step
            train_loss.backward()
            model_optimizer.step()

            # Validation step for non-meta-optimization
            if (epoch + 1) % update_freq != 0:
                model.eval()
                model.set_initial_state(batch_size=len(xs_val))
                state = model.get_state(detach=True)
                val_preds = model(xs_val, state, batch_first=True)[0]
                val_preds = apply_mask(val_preds, xs_val)

                val_loss = loss_fn(
                    val_preds.reshape(-1, model._n_actions),
                    torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1), )

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
