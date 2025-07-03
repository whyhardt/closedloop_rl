import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler

import higher # For differentiable optimization
import time
import numpy as np
from resources.rnn import BaseRNN, CustomEmbedding
from resources.rnn_utils import DatasetRNN

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
    # TODO: Figure out num_workers
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=0)
    if dataset_val is not None:
        # TODO: Find out if Shuffle=True or False? (False was standard)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=0)
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

    # Set up model and optimizer TODO: check if this is necessary
    model = model
    model_optimizer = model_optimizer # TODO: if model_optimizer is None:
    loss_fn = nn.CrossEntropyLoss() # TODO: may need to do this as a module? 
    
    # Define some maybe useless variables TODO: check which are necessary
    continue_training = True
    converged = False
    n_calls_to_train_model = 0
    convergence_value = 1
    last_loss = 1
    recency_factor = 0.5

    train_loss = 0
    val_loss = 0
    iterations_per_epoch = max(len(dataset_train), 64) // batch_size
    save_at_epoch = warmup_steps

    # Histories for plotting
    history_train_loss = []
    history_val_loss = []
    history_log_lambda = []
    history_hypergrad = []

    # Set up meta-optimization
    initial_log_lambda = -5.0  # Initial value for log_lambda
    lr_log_lambda = 1e-3  # Learning rate for log_lambda
    momentum_log_lambda = 0.0  # Momentum for log_lambda TODO: Find out what values mean and try out some
    log_lambda = torch.tensor(initial_log_lambda, requires_grad=True, device=model.device)
    lambda_optimizer = optim.SGD([log_lambda], lr=lr_log_lambda, momentum=momentum_log_lambda)

    # Training loop
    try:
        for epoch in range(epochs):
                # Meaasure the time
            t_start = time.time()

            # Set up the data:
            # Training set TODO: make sure dataloaders work in the loop here
            xs,ys = next(iter(dataloader_train))
            xs, ys = xs.to(model.device), ys.to(model.device)
            # Validation set
            xs_val, ys_val = next(iter(dataloader_val))
            xs_val, ys_val = xs_val.to(model.device), ys_val.to(model.device)

            # Set model to training mode TODO: Find out what this does
            model.train()
            lambda_val = torch.exp(log_lambda)

            # Start the inner loop for meta-optimization
            with higher.innerloop_ctx(model, model_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                # Set initial state TODO: Why
                model.set_initial_state(batch_size=len(xs))
                state = model.get_state(detach=True)

                # Create Mask TODO: Finde out what this does
                mask = xs[..., :1] > -1
                
                # T1 step
                train_preds = fmodel(xs, state, batch_first=True)[0]

                # Add mask to preds
                train_preds = train_preds[0] * mask

                # Calculate L2 TODO: May work with pytoch functions 
                l2 = sum((p ** 2).sum() for p in fmodel.parameters())
                num_params = sum(p.numel() for p in fmodel.parameters())
                l2_norm = l2 / num_params

                train_loss = loss_fn(
                    train_preds.reshape(-1, model._n_actions),
                    torch.argmax(ys.reshape(-1, model._n_actions), dim=1),) + lambda_val * l2_norm
                # Step with the inner loop optimizer
                diffopt.step(train_loss)

                # T2 step
                # Predict and compute val loss
                state = fmodel.get_state(detach=True)
                mask_val = xs_val[..., :1] > -1
                val_preds = fmodel(xs_val, state, batch_first=True)[0]
                val_preds = val_preds * mask_val

                val_loss = loss_fn(
                    val_preds.reshape(-1, model._n_actions),
                    torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1), ) # No regularrization here

                # Recompute train loss for fresh graph after diffopt.step TODO: Imporeve THIS comment
                state = fmodel.get_state(detach=True)
                train_preds = fmodel(xs, state, batch_first=True)[0]
                l2 = sum((p ** 2).sum() for p in fmodel.parameters())
                num_params = sum(p.numel() for p in fmodel.parameters())
                l2_norm = l2 / num_params
                # Do I have to mask again? TODO: Test both versions
                mask = xs[..., :1] > -1
                train_preds = train_preds[0] * mask
                train_loss_updated = loss_fn(train_preds.reshape(-1, model._n_actions), 
                                             torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                             ) + lambda_val * l2_norm

                # Gradients w.r.t. model params
                train_grads = torch.autograd.grad(train_loss_updated, fmodel.parameters(), create_graph=True)
                val_grads = torch.autograd.grad(val_loss, fmodel.parameters(), retain_graph=True)

                # Dot product #?
                dot = sum((v * t).sum() for v, t in zip(val_grads, train_grads))
                # Hypergrad: dot product (?) w.r.t. log_lambda
                hypergrad = torch.autograd.grad(dot, log_lambda)[0]

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
            outputs = model(xs, state, batch_first=True)[0]
            # Using torch sum function here TODO: why?
            l2 = sum(torch.sum(p**2) for p in model.parameters())
            num_params = sum(p.numel() for p in fmodel.parameters())
            l2_norm = l2 / num_params
            # Mask again TODO: Do I have to?
            mask = xs[..., :1] > -1
            outputs = outputs * mask
            # Detach lambda_val from this computation graph so this update does not influence hypergrad calculation
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                 torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                 ) + lambda_val.detach() * l2_norm

            # Backprop step
            train_loss.backward()
            model_optimizer.step()

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
                msg += f"; Time: {time.time()-t_start:.2f}"
                print(msg)
                # TODO: Add convergence back here
                # TODO: Add scheduler lr info here

            # Scheduler update TODO: Add implementation for other schedulers back, check if this sitll works on correct epochs
            if scheduler is not None:
                if epoch < warmup_steps:
                    scheduler_warmup.step()
                else:
                    scheduler.step()
                
            # Save checkpoint TODO: Test if this still works on correct epochs
            if path_save_checkpoints and (epoch+1) == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace(".", f"_ep{epoch+1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        msg = "Training interrupted. Continuing with further operations..."
        
        # Wrap histories 
        histories = [history_train_loss, history_val_loss, history_log_lambda, history_hypergrad]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_val_loss, history_log_lambda, history_hypergrad]
    return model, model_optimizer, histories
