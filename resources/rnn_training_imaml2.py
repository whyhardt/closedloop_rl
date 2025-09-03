import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from resources.rnn import BaseRNN, CustomEmbedding
from resources.rnn_utils import DatasetRNN

# higher and hypergrad libs
import higher
import hypergrad as hg
    
def apply_mask(preds, ss):
    mask = ss[..., :1] > -1
    return preds * mask

class iMAMLTask:
    """
    Handles the train and validation loss for a single task with iMAML regularization
    """
    def __init__(self, model: BaseRNN, reg_params, train_data, val_data, loss_fn):
        device = model.device
        
        # Create stateless version of model for higher-order gradients
        self.fmodel = higher.monkeypatch(model, device=device, copy_initial_weights=True)
        
        self.reg_params = reg_params  # Per-parameter regularization strengths
        self.train_xs, self.train_ys = train_data
        self.val_xs, self.val_ys = val_data
        self.loss_fn = loss_fn
        self.model = model
        
        self.val_loss_value = None
        self.train_loss_value = None
        
    def meta_reg_f(self, meta_params, task_params):
        """
        L2 biased regularization between meta-parameters (bias) and task parameters
        """
        reg_loss = 0.0
        param_idx = 0
        damping = 1e-3
        
        for meta_p, task_p in zip(meta_params, task_params):
            if meta_p.requires_grad and task_p.requires_grad:
                # Get corresponding regularization parameter
                if param_idx < len(self.reg_params):
                    reg_strength = self.reg_params[param_idx] + damping
                    # L2 distance between bias and task parameters
                    reg_loss += reg_strength * ((meta_p - task_p) ** 2).sum()
                    param_idx += 1
        return reg_loss
    
    def train_loss_f(self, task_params, meta_params):
        """
        Training loss with regularization
        """
        # Forward pass with task-specific parameters
        self.model.set_initial_state(batch_size=len(self.train_xs))
        state = self.model.get_state(detach=True)
        
        outputs = self.fmodel(self.train_xs, state, batch_first=True, params=task_params)[0]
        outputs = apply_mask(outputs, self.train_xs)
        
        # Cross-entropy loss
        ce_loss = self.loss_fn(
            outputs.reshape(-1, self.model._n_actions),
            torch.argmax(self.train_ys.reshape(-1, self.model._n_actions), dim=1)
        )
        
        # Add regularization
        reg_loss = self.meta_reg_f(meta_params, task_params)
        
        total_loss = ce_loss + reg_loss
        self.train_loss_value = total_loss.item()
        
        return total_loss
    
    def val_loss_f(self, task_params, meta_params):
        """
        Validation loss (no regularization)
        """
        # Forward pass with task-specific parameters
        self.model.set_initial_state(batch_size=len(self.val_xs))
        state = self.model.get_state(detach=True)
        
        val_outputs = self.fmodel(self.val_xs, state, batch_first=True, params=task_params)[0]
        val_outputs = apply_mask(val_outputs, self.val_xs)
        
        # Val loss with no regularization
        val_loss = self.loss_fn(
            val_outputs.reshape(-1, self.model._n_actions),
            torch.argmax(self.val_ys.reshape(-1, self.model._n_actions), dim=1)
        )
        
        self.val_loss_value = val_loss.item()
        return val_loss

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
    
    meta_update_interval=20,
    inner_steps=3,
    hypergradient_steps=3,
    outer_lr=1e-3,
    initial_reg_param=1e-1,
    ):

    """
    Fit the model with iMAML meta-optimization and per-parameter regularization
    """ 
    if dataset_val is None:
        raise ValueError("Validation dataset is required for training with meta-optimization.")
    
    # Set batch size
    if batch_size == -1:
        batch_size = len(dataset_train)

    # Set bagging
    if bagging:
        batch_size = max(batch_size, 64)  # Ensure batch size is at least 64 for bagging
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size)
    else:
        raise NotImplementedError("Bagging=False is not implemented in this function.")

    # Leave at 0 for compatibility
    num_workers = 0
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Set up learning rate scheduler
    warmup_steps = 1024 if epochs > 1024 else 1
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
                return 1.0 
        
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(model_optimizer, lr_lambda=warmup_lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=model_optimizer, T_0=warmup_steps if warmup_steps > 0 else 64, T_mult=2)
    else:
        scheduler_warmup, scheduler = None, None

    # Epoch error handling
    if epochs == 0:
        raise ValueError("Epochs must be specified.")

    loss_fn = nn.CrossEntropyLoss()
    
    # Define convergence variables
    convergence_value = 1
    last_loss = 1
    recency_factor = 0.5

    # Metaopt setup
    # Initialize per-parameter regularization parameters
    # IMPORTANT: These are kept separate and will NOT be saved with the model
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    reg_params = torch.full((n_params,), initial_reg_param, 
                           device=model.device, requires_grad=True)
    
    # Optimizer for regularization parameters (also not saved)
    reg_optimizer = torch.optim.Adam([reg_params], lr=outer_lr)

    # Initialize losses and histories for logging/plotting
    train_loss = torch.tensor(0.)
    val_loss = torch.tensor(0.)

    history_train_loss = []
    history_val_loss = []
    history_hparams = []  # Store complete regularization parameter tensors

    # Set up at which epoch to save
    save_at_epoch = warmup_steps

    # Inner optimizer setup
    inner_opt_class = hg.GradientDescent
    inner_lr = 1e-3
    inner_opt_kwargs = {'step_size': inner_lr}

    def get_inner_opt(train_loss_f):
        return inner_opt_class(train_loss_f, **inner_opt_kwargs)

    # Training loop
    try:
        for epoch in range(epochs):
            t_epoch_start = time.time()
            train_iter = iter(dataloader_train)
            val_iter = iter(dataloader_val)

            # Regular training step (inner loop)
            try:
                xs, ys = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader_train)
                xs, ys = next(train_iter)

            xs, ys = xs.to(model.device), ys.to(model.device)

            model.train()
            model.set_initial_state(batch_size=len(xs))
            state = model.get_state(detach=True)
            outputs = model(xs, state, batch_first=True)[0]
            outputs = apply_mask(outputs, xs)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                torch.argmax(ys.reshape(-1, model._n_actions), dim=1))
            
            train_loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model_optimizer.step()
            model_optimizer.zero_grad()

            # Meta-update every meta_update_interval epochs
            if (epoch + 1) % meta_update_interval == 0:
                # Get validation data
                try:
                    xs_val, ys_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(dataloader_val)
                    xs_val, ys_val = next(val_iter)
                    
                xs_val, ys_val = xs_val.to(model.device), ys_val.to(model.device)

                # Create iMAML task
                task = iMAMLTask(
                    model, reg_params, 
                    (xs, ys), (xs_val, ys_val), 
                    loss_fn
                )
                
                # Inner optimization
                inner_opt = get_inner_opt(task.train_loss_f)
                
                # Initialize task-specific parameters - must be fresh copies
                task_params = [p.detach().clone().requires_grad_(True) 
                              for p in model.parameters() if p.requires_grad]
                
                # Run inner loop
                params_history = [task_params]
                for inner_step in range(inner_steps):
                    new_params = inner_opt(params_history[-1], list(model.parameters()))
                    # Gradient clipping for inner loop parameters
                    torch.nn.utils.clip_grad_norm_(new_params, max_norm=1.0)
                    params_history.append(new_params)
                
                # Only proceed if inner optimization worked
                if len(params_history) > 1:
                    final_task_params = params_history[-1]
                    
                    # Clear gradients before hypergradient computation
                    reg_optimizer.zero_grad()
                    
                    try:          
                        # Create a combined list of outer parameters including reg_params
                        outer_params = list(model.parameters()) + [reg_params]
                        
                        # Define fixed point map
                        cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=inner_lr)
                        
                        # Now CG should handle reg_params too
                        hg_result = hg.CG(
                            final_task_params, 
                            outer_params,  # Include reg_params here!
                            K=hypergradient_steps, 
                            fp_map=cg_fp_map, 
                            outer_loss=task.val_loss_f
                        )
                        
                        # Check if reg_params got gradients from CG
                        if reg_params.grad is not None:
                            if torch.isnan(reg_params.grad).any():
                                print("WARNING: NaN gradients detected for reg_params. Skipping meta-update this epoch.")
                            else:
                                torch.nn.utils.clip_grad_norm_([reg_params], max_norm=0.5)
                                reg_optimizer.step()
                        else:
                            print("WARNING: reg_params did not receive gradients from CG. Skipping meta-update this epoch.")
                        # Clamp regularization parameters to prevent negative/extreme values
                        with torch.no_grad():
                            reg_params.clamp_(min=1e-4, max=5.0)
                        # Zero out model parameter gradients after metaoptimization
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.zero_()
                    except Exception as e:
                        print(f"ERROR: CG computation failed at epoch {epoch+1}: {e}")
                        print("Skipping meta-update this epoch due to CG failure.")

            else:
                # Compute validation loss for logging without meta-update
                try:
                    xs_val, ys_val = next(val_iter)
                except StopIteration:
                    val_iter = iter(dataloader_val)
                    xs_val, ys_val = next(val_iter)
                    
                xs_val, ys_val = xs_val.to(model.device), ys_val.to(model.device)

                model.eval()
                with torch.no_grad():
                    model.set_initial_state(batch_size=len(xs_val))
                    state = model.get_state(detach=True)
                    val_preds = model(xs_val, state, batch_first=True)[0]
                    val_preds = apply_mask(val_preds, xs_val)
                    val_loss = loss_fn(
                        val_preds.reshape(-1, model._n_actions),
                        torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1)
                    )

            # Update histories
            history_train_loss.append(train_loss.item())
            history_val_loss.append(val_loss.item())
            history_hparams.append(reg_params.detach().cpu().clone())  # Store complete tensor

            # Convergence check
            dloss = last_loss - val_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = val_loss 

            # Logging
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(Val): {val_loss:.4f}"
                msg += f"; Reg Params (min/mean/max): {reg_params.min().item():.4f}/{reg_params.mean().item():.4f}/{reg_params.max().item():.4f}"
                msg += f"; Time: {time.time()-t_epoch_start:.2f}"

                if converged:
                    msg += " --- Converged!"
                print(msg)

            # Learning rate scheduling
            if scheduler is not None:
                if epoch < warmup_steps:
                    scheduler_warmup.step()
                else:
                    scheduler.step()
                
            # Save checkpoint - ONLY model state_dict, NO reg_params
            if path_save_checkpoints and (epoch + 1) == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace(".", f"_ep{epoch + 1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        print("Training interrupted. Continuing with further operations...")

        histories = [history_train_loss, history_val_loss, history_hparams]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_val_loss, history_hparams]
    return model, model_optimizer, histories