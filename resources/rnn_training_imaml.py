import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from resources.rnn import BaseRNN, CustomEmbedding
from resources.rnn_utils import DatasetRNN

def compute_hvp(loss, params, vector, damping=1e-6):
    """Compute Hessian-vector product"""
    # Compute first-order gradients
    grads = torch.autograd.grad(
        loss, params, 
        create_graph=True, 
        retain_graph=True, 
        allow_unused=True
    )
    
    # Filter valid gradients
    valid_grads = [g for g in grads if g is not None]
    valid_params = [p for g, p in zip(grads, params) if g is not None]
    
    if len(valid_grads) == 0:
        return torch.zeros_like(vector)
    
    # Create flat gradient
    flat_grad = torch.cat([g.view(-1) for g in valid_grads])
    
    # Extract corresponding vector parts
    valid_vector_parts = []
    vector_idx = 0
    for g, p in zip(grads, params):
        param_size = p.numel()
        if g is not None:
            valid_vector_parts.append(vector[vector_idx:vector_idx + param_size])
        vector_idx += param_size
    
    valid_vector = torch.cat(valid_vector_parts)
    
    # Compute gradient-vector product
    grad_product = flat_grad.dot(valid_vector)
    
    # Compute second-order gradients (HVP)
    hvp_grads = torch.autograd.grad(
        grad_product, valid_params, 
        retain_graph=True,
        allow_unused=True
    )
    
    # Process HVP gradients
    valid_hvp_grads = [h if h is not None else torch.zeros_like(p) 
                      for h, p in zip(hvp_grads, valid_params)]
    
    # Flatten HVP gradients
    flat_hvp_valid = torch.cat([h.view(-1) for h in valid_hvp_grads])
    
    # Reconstruct full HVP vector
    full_hvp = torch.zeros_like(vector)
    vector_idx = 0
    valid_idx = 0
    
    for g, p in zip(grads, params):
        param_size = p.numel()
        if g is not None:
            hvp_size = valid_hvp_grads[valid_idx].numel()
            if hvp_size == param_size:
                hvp_start = sum(h.numel() for h in valid_hvp_grads[:valid_idx])
                hvp_end = hvp_start + param_size
                if hvp_end <= len(flat_hvp_valid):
                    full_hvp[vector_idx:vector_idx + param_size] = flat_hvp_valid[hvp_start:hvp_end]
            valid_idx += 1
        vector_idx += param_size
    
    # Apply damping
    return full_hvp + damping * vector

def conjugate_gradient(Av_func, b, x_init=None, max_iter=10, tol=1e-10, damping=1e-8):
    """Conjugate gradient solver with essential fallbacks only"""
    def _is_healthy(tensor):
        return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())
    
    x = torch.zeros_like(b) if x_init is None else x_init.clone()
    problem_scale = b.norm().item()
    
    # Early termination if b is very small
    if problem_scale < tol:
        return x
    
    base_damping = damping * max(1.0, problem_scale)
    
    # Try standard CG with progressive damping
    for damping_attempt in range(3):
        current_damping = base_damping * (10 ** damping_attempt)
        
        x_attempt = x.clone()
        r = b - Av_func(x_attempt)
        p = r.clone()
        rsold = r.dot(r)
        
        success = True
        
        for i in range(max_iter):
            Ap = Av_func(p)
            
            if not _is_healthy(Ap):
                success = False
                break
            
            pAp = p.dot(Ap) + current_damping * p.dot(p)
            
            # Handle small denominators
            if torch.abs(pAp) < current_damping * 0.01:
                emergency_damping = current_damping * 100
                pAp_emergency = p.dot(Ap) + emergency_damping * p.dot(p)
                if torch.abs(pAp_emergency) < emergency_damping * 0.01:
                    success = False
                    break
                pAp = pAp_emergency
                current_damping = emergency_damping
            
            alpha = rsold / pAp
            
            if not _is_healthy(alpha) or torch.abs(alpha) > 1e3:
                success = False
                break
            
            x_attempt = x_attempt + alpha * p
            r = r - alpha * Ap
            rsnew = r.dot(r)
            
            # Check convergence
            if rsnew.sqrt() < tol * 10:
                return x_attempt
            
            beta = rsnew / (rsold + current_damping * 1e-6)
            
            if not _is_healthy(beta):
                success = False
                break
            
            p = r + beta * p
            rsold = rsnew
        
        if success:
            return x_attempt
    
    # Fallback: steepest descent
    print("Conjugate gradient failed, using steepest descent fallback")
    learning_rate = 0.01 / (1.0 + problem_scale)
    x_descent = x.clone()
    
    for i in range(min(max_iter, 5)):
        residual = b - Av_func(x_descent)
        if not _is_healthy(residual):
            break
        x_descent = x_descent + learning_rate * residual
        if residual.norm() < tol * 100:
            return x_descent
    
    return x_descent

class iMAMLOptimizer:
    def __init__(self, meta_lr=0.001, cg_steps=5, cg_damping=1e-2, init_lambda=1.0):
        self.meta_lr = meta_lr
        self.cg_steps = cg_steps
        self.cg_damping = cg_damping
        self.min_lambda = 1e-6
        self.max_lambda = 1e6
        
        log_lambda_init = np.log(max(init_lambda, self.min_lambda))
        self.log_l2_lambda = nn.Parameter(torch.tensor(log_lambda_init))

    @property
    def l2_lambda(self):
        return torch.clamp(torch.exp(self.log_l2_lambda), self.min_lambda, self.max_lambda)
    
    def compute_lambda_meta_gradient(self, model, inner_loss, outer_loss):
        """Compute meta-gradient using implicit differentiation for lambda"""
        params = list(model.parameters())
        
        # Get outer gradients
        outer_grads = torch.autograd.grad(
            outer_loss, params, 
            create_graph=False, 
            retain_graph=True, 
            allow_unused=True
        )
        
        # Filter valid gradients
        valid_grads = [g for g in outer_grads if g is not None]
        valid_params = [p for g, p in zip(outer_grads, params) if g is not None]
        param_indices = [i for i, g in enumerate(outer_grads) if g is not None]
        
        if len(valid_grads) == 0:
            return torch.tensor(0.0)
        
        flat_outer_grad = torch.cat([g.view(-1) for g in valid_grads])
        
        # Define A(v) = v + H*v/λ
        def Av_func(v):
            hvp = compute_hvp(inner_loss, valid_params, v, damping=self.cg_damping)
            return v + hvp / self.l2_lambda
        
        # Solve A*x = outer_grad
        implicit_grad = conjugate_gradient(
            Av_func, 
            flat_outer_grad, 
            max_iter=self.cg_steps,
            tol=1e-8,
            damping=self.cg_damping
        )
        
        # Compute parameter difference from initial values
        param_diff_parts = []
        for i in param_indices:
            if hasattr(model, 'initial_params') and i < len(model.initial_params):
                param_diff_parts.append((params[i] - model.initial_params[i]).view(-1))
        
        if len(param_diff_parts) == 0:
            return torch.tensor(0.0)
        
        param_diff = torch.cat(param_diff_parts)
        
        # Compute ∂L_val/∂λ = ∂L_val/∂φ * ∂φ/∂λ
        dL_dlambda = -implicit_grad.dot(param_diff) / self.l2_lambda
        
        # Account for log parameterization
        meta_grad = dL_dlambda * self.l2_lambda
        
        return meta_grad
    
    def update_lambda(self, meta_grad):
        """Update the L2 weight decay parameter"""
        with torch.no_grad():
            self.log_l2_lambda -= self.meta_lr * meta_grad
    
    def compute_theta_meta_grad(self, model, inner_loss, outer_loss):
        """Compute meta-gradient w.r.t. θ using implicit differentiation"""
        params = list(model.parameters())
        
        # Get outer gradients
        outer_grads = torch.autograd.grad(
            outer_loss, params, 
            create_graph=False, 
            retain_graph=True, 
            allow_unused=True
        )
        
        # Filter valid gradients
        valid_grads = [g for g in outer_grads if g is not None]
        valid_params = [p for g, p in zip(outer_grads, params) if g is not None]
        
        if len(valid_grads) == 0:
            return [torch.zeros_like(p) for p in params]
        
        flat_outer_grad = torch.cat([g.view(-1) for g in valid_grads])
        
        # Define A(v) = v + H*v/λ
        def Av_func(v):
            hvp = compute_hvp(inner_loss, valid_params, v, damping=self.cg_damping)
            return v + hvp / self.l2_lambda
        
        # Solve A*x = outer_grad
        implicit_grad_flat = conjugate_gradient(
            Av_func, 
            flat_outer_grad, 
            max_iter=self.cg_steps,
            tol=1e-8,
            damping=self.cg_damping
        )
        
        # Map back to parameter structure
        meta_grad = []
        valid_idx = 0
        
        for g, p in zip(outer_grads, params):
            if g is not None:
                param_size = p.numel()
                grad_start = sum(valid_params[j].numel() for j in range(valid_idx))
                grad_end = grad_start + param_size
                
                if grad_end <= len(implicit_grad_flat):
                    grad_chunk = implicit_grad_flat[grad_start:grad_end].view_as(p)
                    meta_grad.append(grad_chunk)
                else:
                    meta_grad.append(torch.zeros_like(p))
                valid_idx += 1
            else:
                meta_grad.append(torch.zeros_like(p))
        
        return meta_grad
    
    def update_theta(self, model, meta_grad):
        """Apply meta-gradient update"""
        with torch.no_grad():
            for p, g in zip(model.parameters(), meta_grad):
                if p.requires_grad:
                    p -= self.meta_lr * g
    
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
    meta_update_interval: int = 10,  # <-- NEW: how often to update lambda
    imaml_optimizer: iMAMLOptimizer = iMAMLOptimizer,  # <-- NEW: pass your optimizer
    ):

    """
    Fit the model with iMAML meta-optimization

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
        imaml_optimizer: instance of iMAMLOptimizer for meta-learning

    Returns:
        model: trained model
        model_optimizer: optimizer used for training
        histories: list of training and validation losses, theta drift, and meta-gradient norms
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
    
    num_workers = 0
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        raise ValueError("Validation dataset is required for training with meta-optimization.")


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
                
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=model_optimizer, T_0=warmup_steps if warmup_steps > 0 else 64, T_mult=2)
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

    # Inititalize losses and histories for logging/plotting
    train_loss = torch.tensor(0.)
    val_loss = torch.tensor(0.)

    meta_grad = torch.tensor(0.)
    meta_grad_norm = 0.0

    avg_param_norm = torch.tensor(0.)

    history_train_loss = []
    history_val_loss = []
    history_theta = []
    history_metagrad = []

    # Set up at which epoch to save
    save_at_epoch = warmup_steps

    # Training loop
    try:
        for epoch in range(epochs):
            t_epoch_start = time.time()
            train_iter = iter(dataloader_train)
            val_iter = iter(dataloader_val)

            # Save initial parameters for implicit differentiation
            model.initial_params = [p.detach().clone() for p in model.parameters()]

            xs, ys = next(train_iter)
            xs, ys = xs.to(model.device), ys.to(model.device)

            # INNER LOOP, normal training step
            model.train()
            model.set_initial_state(batch_size=len(xs))
            state = model.get_state(detach=True)
            outputs = model(xs, state, batch_first=True)[0]
            outputs = apply_mask(outputs, xs)
            main_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                 torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                 )
            
            reg = 0
            for p, p0 in zip(model.parameters(), model.initial_params):
                reg += torch.sum(p - p0) ** 2
            train_loss = main_loss + 0.5 * imaml_optimizer.l2_lambda * reg

            model_optimizer.zero_grad()
            train_loss.backward()
            model_optimizer.step()

            # OUTER LOOP, Meta-update every meta_update_interval
            xs_val, ys_val = next(val_iter)
            xs_val, ys_val = xs_val.to(model.device), ys_val.to(model.device)

            # Compute validation loss
            model.set_initial_state(batch_size=len(xs_val))
            state = model.get_state(detach=True)
            val_preds = model(xs_val, state, batch_first=True)[0]
            val_preds = apply_mask(val_preds, xs_val)
            val_loss = loss_fn(
                val_preds.reshape(-1, model._n_actions),
                torch.argmax(ys_val.reshape(-1, model._n_actions), dim=1),
            )

            if imaml_optimizer is not None and (epoch + 1) % meta_update_interval == 0:

                # Recompute train loss with grads for meta-grad computation
                model.train()
                model.set_initial_state(batch_size=len(xs))
                state = model.get_state(detach=True)
                outputs = model(xs, state, batch_first=True)[0]
                outputs = apply_mask(outputs, xs)

                main_outer_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                    torch.argmax(ys.reshape(-1, model._n_actions), dim=1),
                                    )
                
                reg = 0
                for p, p0 in zip(model.parameters(), model.initial_params):
                    reg += torch.sum(p - p0) ** 2
                train_outer_loss = main_outer_loss + 0.5 * imaml_optimizer.l2_lambda * reg

                # Compute meta-gradient for λ
                # meta_grad_lambda = imaml_optimizer.compute_lambda_meta_gradient(
                #     model, train_outer_loss, val_loss
                # )

                # # Update lambda TODO: Find out if I want to update lambda at all in iMAML
                # imaml_optimizer.update_lambda(meta_grad_lambda)

                # Compute meta-gradient for θ
                meta_grad = imaml_optimizer.compute_theta_meta_grad(
                    model, train_outer_loss, val_loss
                )

                # Update theta TODO: Find out if I want to update theta at all in iMAML
                imaml_optimizer.update_theta(model, meta_grad)

                # Log the L2 norm of the meta-gradient
                meta_grad_flat = torch.cat([g.view(-1) for g in meta_grad if g is not None])
                meta_grad_norm = meta_grad_flat.norm().item()

            # Compute average parameter norm for monitoring
            avg_param_norm = np.mean([p.norm().item() for p in model.parameters()])

            history_train_loss.append(train_loss.item())
            history_val_loss.append(val_loss.item())
            history_theta.append(avg_param_norm)
            history_metagrad.append(meta_grad_norm)

            dloss = last_loss - val_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = val_loss 

            msg = None
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(Val): {val_loss:.4f}"
                msg += f"; Average Parameter Norm: {avg_param_norm:.2f}"
                msg += f"; Meta-Gradient Norm: {meta_grad_norm:.4f}"
                msg += f"; Time: {time.time()-t_epoch_start:.2f}"

                if converged:
                    msg += " --- Converged!"
                print(msg)

            if scheduler is not None:
                if epoch < warmup_steps:
                    scheduler_warmup.step()
                else:
                    scheduler.step()
                
            # Save checkpoint, also keep optimizer state
            if path_save_checkpoints and (epoch + 1) == save_at_epoch:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': model_optimizer.state_dict(),
                    'epoch': epoch + 1,
                }
                torch.save(checkpoint, path_save_checkpoints.replace(".", f"_ep{epoch + 1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        msg = "Training interrupted. Continuing with further operations..."
        
        histories = [history_train_loss, history_val_loss, history_theta, history_metagrad]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_val_loss, history_theta, history_metagrad]
    return model, model_optimizer, histories
