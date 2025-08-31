import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from resources.rnn import BaseRNN, CustomEmbedding
from resources.rnn_utils import DatasetRNN

try:
    import higher
    import hypergrad as hg
except ImportError:
    raise ImportError("Please install 'higher' and 'hypergrad' packages for iMAML functionality")
    
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
        
    def bias_reg_f(self, bias_params, task_params):
        """L2 biased regularization between meta-parameters (bias) and task parameters"""
        reg_loss = 0.0
        param_idx = 0
        
        for bias_p, task_p in zip(bias_params, task_params):
            if bias_p.requires_grad and task_p.requires_grad:
                # Get corresponding regularization parameter
                if param_idx < len(self.reg_params):
                    reg_strength = self.reg_params[param_idx]
                    # L2 distance between bias and task parameters
                    reg_loss += reg_strength * ((bias_p - task_p) ** 2).sum()
                    param_idx += 1
                
        return reg_loss
    
    def train_loss_f(self, task_params, meta_params):
        """Training loss with biased regularization"""
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
        
        # Add biased regularization
        reg_loss = self.bias_reg_f(meta_params, task_params)
        
        total_loss = ce_loss + reg_loss
        self.train_loss_value = total_loss.item()
        
        return total_loss
    
    def val_loss_f(self, task_params, meta_params):
        """Validation loss (no regularization)"""
        # Forward pass with task-specific parameters
        self.model.set_initial_state(batch_size=len(self.val_xs))
        state = self.model.get_state(detach=True)
        
        val_outputs = self.fmodel(self.val_xs, state, batch_first=True, params=task_params)[0]
        val_outputs = apply_mask(val_outputs, self.val_xs)
        
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
    
    meta_update_interval=5,
    inner_steps=3,
    hypergradient_steps=5,
    outer_lr=0.1,
    initial_reg_param=0.01,
    ):

    """
    Fit the model with iMAML meta-optimization and per-parameter regularization

    Args:
        model: the model to train
        dataset_train: training dataset
        dataset_val: validation dataset (required for meta-optimization)
        model_optimizer: optimizer for the model parameters
        convergence_threshold: threshold for convergence
        epochs: number of training epochs
        batch_size: size of each training batch, -1 for full dataset (-1 used for thesis)
        bagging: whether to use bagging
        scheduler: whether to use a learning rate scheduler (CosineAnnealingWarmRestarts used in thesis)
        n_steps: number of steps for training, -1 for full dataset
        verbose: whether to print training progress
        path_save_checkpoints: path to save model checkpoints
        meta_update_interval: At which frequency to run an iMAML meta-update step
        inner_steps: Number of inner optimization steps
        hypergradient_steps: Number of CG steps for hypergradient computation
        outer_lr: Learning rate for outer optimization
        initial_reg_param: Initial regularization parameter value

    Returns:
        model: trained model
        model_optimizer: optimizer used for training
        histories: list of training and validation losses, regularization parameters stats
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

    # Initialize per-parameter regularization parameters
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    reg_params = torch.full((n_params,), initial_reg_param, 
                           device=model.device, requires_grad=True)
    
    # Optimizer for regularization parameters
    reg_optimizer = torch.optim.Adam([reg_params], lr=outer_lr)

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

    # Initialize losses and histories for logging/plotting
    train_loss = torch.tensor(0.)
    val_loss = torch.tensor(0.)

    meta_grad_norm = 0.0
    avg_param_norm = torch.tensor(0.)

    history_train_loss = []
    history_val_loss = []
    history_hparams = []  # Store complete regularization parameter tensors
    history_meta_grad_norm = []

    # Set up at which epoch to save
    save_at_epoch = warmup_steps

    # Inner optimizer setup
    inner_opt_class = hg.GradientDescent
    inner_lr = 0.1
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
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                torch.argmax(ys.reshape(-1, model._n_actions), dim=1))
            
            train_loss.backward()
            model_optimizer.step()
            model_optimizer.zero_grad()

            # Meta-update every meta_update_interval epochs
            if (epoch + 1) % meta_update_interval == 0:  # Fixed timing: +1 so we get epochs 5, 10, 15...
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
                    try:
                        new_params = inner_opt(params_history[-1], list(model.parameters()))
                        params_history.append(new_params)
                    except Exception as e:
                        print(f"Inner optimization failed at step {inner_step}: {e}")
                        break
                
                if len(params_history) > 1:  # Only proceed if inner optimization worked
                    final_task_params = params_history[-1]
                    
                    # Clear gradients before hypergradient computation
                    reg_optimizer.zero_grad()
                    
                    print(f"\n=== META-UPDATE at epoch {epoch+1} ===")
                    print(f"Reg params before update: min={reg_params.min().item():.6f}, mean={reg_params.mean().item():.6f}, max={reg_params.max().item():.6f}")
                    
                    try:
                        # The issue: CG doesn't know about reg_params! 
                        # We need to include reg_params in the outer parameters for CG
                        print("Starting CG computation...")
                        
                        # Create a combined list of outer parameters including reg_params
                        outer_params = list(model.parameters()) + [reg_params]
                        
                        cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=inner_lr)
                        
                        # Now CG should handle reg_params too
                        hg_result = hg.CG(
                            final_task_params, 
                            outer_params,  # Include reg_params here!
                            K=hypergradient_steps, 
                            fp_map=cg_fp_map, 
                            outer_loss=task.val_loss_f
                        )
                        
                        print("CG computation completed successfully")
                        
                        # Check if reg_params got gradients from CG
                        if reg_params.grad is not None:
                            print(f"SUCCESS: CG provided gradients! Grad norm: {reg_params.grad.norm().item():.6f}")
                            print(f"CG grad stats: min={reg_params.grad.min().item():.6f}, mean={reg_params.grad.mean().item():.6f}, max={reg_params.grad.max().item():.6f}")
                            meta_grad_norm = reg_params.grad.norm().item()
                            
                            # Instead of minimizing (which drives params to zero), maybe we should maximize
                            # or use a different update rule. Let's see what happens with normal updates first.
                            reg_optimizer.step()
                            print("Reg params updated via CG gradients")
                            
                        else:
                            print("CG still did not provide gradients, trying manual computation...")
                            # Manual gradient computation 
                            val_loss_value = task.val_loss_f(final_task_params, list(model.parameters()))
                            val_loss_value.backward(retain_graph=True)
                            
                            if reg_params.grad is not None:
                                print(f"Manual gradients obtained! Grad norm: {reg_params.grad.norm().item():.6f}")
                                print(f"Manual grad stats: min={reg_params.grad.min().item():.6f}, mean={reg_params.grad.mean().item():.6f}, max={reg_params.grad.max().item():.6f}")
                                meta_grad_norm = reg_params.grad.norm().item()
                                
                                # The gradients are negative because lower validation loss is better,
                                # and apparently less regularization leads to better validation performance.
                                # This might actually be correct! But let's try a smaller learning rate.
                                with torch.no_grad():
                                    # Manual update with smaller step size to prevent collapse
                                    reg_params -= 0.001 * reg_params.grad  # Much smaller step
                                print("Reg params updated via manual gradients (small step)")
                            else:
                                print(f"ERROR: Still no gradients for reg_params at epoch {epoch+1}")
                                meta_grad_norm = 0.0
                        
                        # Clamp regularization parameters to prevent negative/extreme values
                        old_params = reg_params.clone()
                        with torch.no_grad():
                            reg_params.clamp_(min=1e-6, max=1.0)  # More reasonable upper bound
                        
                        if not torch.equal(old_params, reg_params):
                            print(f"Clamped reg params from range [{old_params.min().item():.6f}, {old_params.max().item():.6f}]")
                        
                        print(f"Reg params after update: min={reg_params.min().item():.6f}, mean={reg_params.mean().item():.6f}, max={reg_params.max().item():.6f}")
                                
                    except Exception as e:
                        print(f"ERROR: CG computation failed at epoch {epoch}: {e}")
                        import traceback
                        traceback.print_exc()
                        meta_grad_norm = 0.0
                    
                    print("=== END META-UPDATE ===\n")
                else:
                    meta_grad_norm = 0.0
                    print(f"Warning: Inner optimization failed at epoch {epoch}")

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

            # Compute average parameter norm
            with torch.no_grad():
                param_norms = [p.norm() for p in model.parameters() if p.requires_grad]
                avg_param_norm = torch.stack(param_norms).mean()

            # Update histories
            history_train_loss.append(train_loss.item())
            history_val_loss.append(val_loss.item())
            history_hparams.append(reg_params.detach().cpu().clone())  # Store complete tensor
            history_meta_grad_norm.append(meta_grad_norm)

            # Convergence check
            dloss = last_loss - val_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = val_loss 

            # Logging
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(Val): {val_loss:.4f}"
                msg += f"; Avg Param Norm: {avg_param_norm:.2f}"
                msg += f"; Meta-Grad Norm: {meta_grad_norm:.4f}"
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
                
            # Save checkpoint
            if path_save_checkpoints and (epoch + 1) == save_at_epoch:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': model_optimizer.state_dict(),
                    'reg_params': reg_params,
                    'reg_optimizer': reg_optimizer.state_dict(),
                    'epoch': epoch + 1,
                }
                torch.save(checkpoint, path_save_checkpoints.replace(".", f"_ep{epoch + 1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        print("Training interrupted. Continuing with further operations...")

    histories = [
        history_train_loss, 
        history_val_loss, 
        history_hparams  # Complete regularization parameter tensors for plotting
    ]
    return model, model_optimizer, histories