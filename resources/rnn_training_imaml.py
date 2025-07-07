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
    update_freq = 35
    initial_log_lambda = -5.0  # Initial value for log_lambda
    lr_log_lambda = 0.1  # Learning rate for log_lambda
    momentum_log_lambda = 0.9  # Momentum for log_lambda TODO: Try out some
    # I am also clipping the hypergrads, its in the loop

    # Lookahead number of steps
    num_inner_steps = 5

    # Conjugate gradient hyperparams
    cgh_damping = 0.1
    cgh_n_steps = 5

    log_lambda = torch.tensor(initial_log_lambda, requires_grad=True, device=model.device)
    lambda_optimizer = optim.SGD([log_lambda], lr=lr_log_lambda, momentum=momentum_log_lambda)

    # Define inner training function:
    def inner_train(model, xs, ys, lr, lambda_val, num_inner_steps=num_inner_steps):
        # Create a copy of the model
        inner = deepcopy(model)
        # Create a new optimizer for the inner model
        opt = optim.SGD(inner.parameters(), lr=lr)

        for _ in range(num_inner_steps):
            # Forward pass
            inner.train()
            inner.set_initial_state(batch_size=len(xs))
            state = inner.get_state(detach=True)
            preds = inner(xs, state, batch_first=True)[0]
            preds = apply_mask(preds, xs)

            # Recompute L2 norm
            params = torch.cat([p.view(-1) for p in inner.parameters()])
            l2_norm = (params ** 2).mean()

            # Compute inner loss
            loss = loss_fn(preds.reshape(-1, model._n_actions),
                        torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                        ) + lambda_val.detach() * l2_norm

            opt.zero_grad()
            loss.backward()
            opt.step()
        return inner
    
    # Method to approximate the inverse Hessian-vector product using conjugate gradients
    def conjugate_gradients(val_grad, params, damping=cgh_damping, n_steps=cgh_n_steps):
        x = [torch.zeros_like(p) for p in params]

        for _ in range(n_steps):
            dot = sum((g * xi).sum() for g, xi in zip(val_grad, x))
            Hv = torch.autograd.grad(dot, params, retain_graph=True)
            x = [x_i - damping * x_i + h for x_i, h in zip(x, Hv)]

        return x


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

            # Inner loop for meta-optimization
            if (epoch + 1) % update_freq == 0:
                # Step 1: Train inner model
                # TODO: Using fixed lr here, adapt to use scheduled one 
                inner_model = inner_train(model, xs, ys, lr=1e-2, lambda_val=lambda_val)

                # Step 2: Compute training grad w.r.t val loss
                # First compute val loss
                inner_model.set_initial_state(batch_size=len(xs_val))
                state = inner_model.get_state(detach=True)
                val_preds = inner_model(xs_val, state, batch_first=True)[0]
                val_preds = apply_mask(val_preds, xs_val)
                val_loss = loss_fn(
                    val_preds.reshape(-1, model._n_actions),
                    torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1), ) # No regularization here
                # Now compute grads
                val_grads = torch.autograd.grad(val_loss, inner_model.parameters(), create_graph=True) # Maybe retain graph
                pass

                # Step 3: Approximate inverse Hessian-vector product
                # Using the conjugate gradient method
                params = [p for p in inner_model.parameters() if p.requires_grad]
                inv_hvp = conjugate_gradients(val_grads, params, damping=1e-2, n_steps=5) # TODO: Find out how this algo works and what damping and n_steps do

                # Step 4: Compute gradient of log_lambda w.r.t. the reg term that could be applied
                reg_term = torch.cat([p.view(-1) for p in inner_model.parameters()]) ** 2
                reg_term = reg_term.mean()  # L2 norm
                # Compute full regularization
                reg = lambda_val * reg_term
                # reg_grad = torch.autograd.grad(reg_term, log_lambda, retain_graph=True)[0]
                reg_grad = reg.detach() 

                # Step 5: Compute the hypergrad
                hypergrad = -sum((g * v_i).sum() for g, v_i in zip(val_grads, inv_hvp)) + reg_grad
                hypergrad = torch.clamp(hypergrad, -5.0, 5.0) # TODO: Find out if clipping is necessary; It's not but keeps from exploding

                # Step 6: Update log_lambda
                lambda_optimizer.zero_grad()
                log_lambda.grad = hypergrad
                lambda_optimizer.step()


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
