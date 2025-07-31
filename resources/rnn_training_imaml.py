import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

import time
import numpy as np
from resources.rnn import BaseRNN, CustomEmbedding
from resources.rnn_utils import DatasetRNN

# Compute conjugate gradient TODO: Understand this
def conjugate_gradient(Av_func, b, x_init=None, max_iter=10, tol=1e-10):
    """
    Conjugate gradient solver for Ax = b

    Args:
        Av_func: function that computes matrix-vector product Av
        b: right hand side vector
        x_init: initial guess for x
    """
    x = torch.zeros_like(b) if x_init is None else x_init.clone()
    r = b - Av_func(x)
    p = r.clone()
    rsold = r.dot(r)
    
    for _ in range(max_iter):
        Ap = Av_func(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r.dot(r)
        
        if rsnew.sqrt() < tol:
            break
            
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    
    return x

# Compute hessian vector product TODO: Understand this
def compute_hvp(loss, params, vector):
    """
    Compute Hessian-vector product H*v where H is the Hessian of loss w.r.t. params

    Args:
        loss: scalar loss value
        params: list of model parameters
        vector: vector to multiply with Hessian
    """
    # Compute gradient
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    flat_grad = torch.cat([g.view(-1) for g in grads])
    
    # Compute Hessian-vector product
    grad_product = flat_grad.dot(vector)
    hvp = torch.autograd.grad(grad_product, params, retain_graph=True)
    flat_hvp = torch.cat([g.view(-1) for g in hvp])
    
    return flat_hvp

class iMAMLOptimizer:
    def __init__(self, meta_lr=0.01, cg_steps=5, cg_damping=1e-5, init_lambda=1.0):
        """
        iMAML optimizer for meta-learning L2 weight decay
        
        Args:
            meta_lr: learning rate for meta-parameter (L2 weight decay)
            cg_steps: number of conjugate gradient steps
            cg_damping: damping factor for CG
        """
        self.meta_lr = meta_lr
        self.cg_steps = cg_steps
        self.cg_damping = cg_damping

        log_lambda_init = np.log(init_lambda)
        
        # Initialize Lambda in iMAML on log scale for possible auto adaptation
        self.log_l2_lambda = nn.Parameter(torch.tensor(log_lambda_init))

    @property # TODO: Understand why this is a property
    def l2_lambda(self):
        return torch.exp(self.log_l2_lambda)
    
    def compute_lambda_meta_gradient(self, model, inner_loss, outer_loss):
        """
        Compute meta-gradient using implicit differentiation
        
        Args:
            model: the model being trained
            inner_loss: training loss
            outer_loss: validation loss
            inner_params: converged parameters from inner optimization
        """
        # Get model parameters
        params = list(model.parameters())
        
        # Compute outer gradient ∇φ L_val(φ*)
        outer_grad = torch.autograd.grad(outer_loss, params, retain_graph=True)
        flat_outer_grad = torch.cat([g.view(-1) for g in outer_grad])
        
        # Define function for (I + λ^(-1) H) v
        def Av_func(v):
            # Compute Hessian-vector product
            hvp = compute_hvp(inner_loss, params, v)
            # Return (I + λ^(-1) H) v
            return v + hvp / self.l2_lambda
        
        # Solve (I + λ^(-1) H) g = ∇φ L_val using CG
        implicit_grad = conjugate_gradient(
            Av_func, 
            flat_outer_grad, 
            max_iter=self.cg_steps,
            tol=1e-10
        )
        
        # The meta-gradient w.r.t. λ requires computing ∂φ*/∂λ
        # For L2 regularization: ∂φ*/∂λ = -(I + λ^(-1) H)^(-1) * (φ* - θ)
        param_diff = torch.cat([
            (p - p0).view(-1) 
            for p, p0 in zip(params, model.initial_params)
        ])
        
        # Compute ∂L_val/∂λ = ∂L_val/∂φ * ∂φ/∂λ
        dL_dlambda = -implicit_grad.dot(param_diff) / self.l2_lambda
        
        # Account for log parameterization
        meta_grad = dL_dlambda * self.l2_lambda
        
        return meta_grad
    
    def update_lambda(self, meta_grad):
        """
        Update the L2 weight decay parameter
        """
        with torch.no_grad():
            self.log_l2_lambda -= self.meta_lr * meta_grad

    def compute_theta_meta_grad(self, model, inner_loss, outer_loss):
        """
        Compute meta-gradient w.r.t. θ using implicit differentiation

        Args:
            model: the model being trained
            inner_loss: training loss
            outer_loss: recomputed training loss
        """
        params = list(model.parameters())

        # Outer gradient: ∇φ L_val(φ)
        outer_grad = torch.autograd.grad(outer_loss, params, retain_graph=True)
        flat_outer_grad = torch.cat([g.view(-1) for g in outer_grad])

        def Av_func(v):
            hvp = compute_hvp(inner_loss, params, v)
            return v + hvp / self.l2_lambda

        # Solve implicit gradient
        implicit_grad_flat = conjugate_gradient(Av_func, flat_outer_grad, max_iter=self.cg_steps)

        # Unflatten back to per-parameter shape
        meta_grad = []
        pointer = 0
        for p in params:
            numel = p.numel()
            meta_grad.append(implicit_grad_flat[pointer:pointer + numel].view_as(p))
            pointer += numel

        return meta_grad

    def update_theta(self, model, meta_grad):
        """
        Apply the meta-gradient to update model parameters θ
        """
        with torch.no_grad():
            for p, g in zip(model.parameters(), meta_grad):
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
    avg_meta_grad = torch.tensor(0.)

    theta_init_snapshot = [p.clone().detach() for p in model.parameters()]
    theta_drift = torch.tensor(0.)
    drift_ratio = torch.tensor(0.)

    history_train_loss = []
    history_val_loss = []
    history_theta_drift = []
    history_metagrad = []

    # Set up at which epoch to save
    save_at_epoch = warmup_steps

    # Training loop
    try:
        for epoch in range(epochs):
            t_epoch_start = time.time()

            # Save initial parameters for implicit differentiation
            model.initial_params = [p.detach().clone() for p in model.parameters()]

            xs, ys = next(iter(dataloader_train))
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
            xs_val, ys_val = next(iter(dataloader_val))
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
                xs,ys = next(iter(dataloader_train))
                xs, ys = xs.to(model.device), ys.to(model.device)

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
                # meta_grad = imaml_optimizer.compute_lambda_meta_gradient(
                #     model, train_outer_loss, val_loss
                # )

                # Update lambda TODO: Find out if I want to update lambda at all in iMAML
                # imaml_optimizer.update_lambda(meta_grad)

                # Compute meta-gradient for θ
                meta_grad = imaml_optimizer.compute_theta_meta_grad(
                    model, train_outer_loss, val_loss
                )

                # Update theta TODO: Find out if I want to update theta at all in iMAML
                imaml_optimizer.update_theta(model, meta_grad)

                # Compute theta drift and average meta-gradient norm for logging/plotting
                theta_drift = sum((p - p0).norm().item() for p, p0 in zip(model.parameters(), theta_init_snapshot))
                avg_param_norm = np.mean([p.norm().item() for p in model.parameters()])
                drift_ratio = theta_drift / avg_param_norm
                avg_meta_grad = torch.mean(torch.stack([g.norm() for g in meta_grad])).item()

            history_train_loss.append(train_loss.item())
            history_val_loss.append(val_loss.item())
            history_theta_drift.append(drift_ratio)
            history_metagrad.append(avg_meta_grad)

            dloss = last_loss - val_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = val_loss 

            msg = None
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(Val): {val_loss:.4f}"
                msg += f"; θ drift ratio: {drift_ratio:.2f}"
                msg += f"; metagrad: {avg_meta_grad:.6f}"
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
        
        histories = [history_train_loss, history_val_loss, history_theta_drift, history_metagrad]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_val_loss, history_theta_drift, history_metagrad]
    return model, model_optimizer, histories
