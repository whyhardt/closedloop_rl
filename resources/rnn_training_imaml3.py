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

# Import SINDy utilities
from resources.imaml_utils import fit_sindy_torch

def apply_mask(preds, ss):
    mask = ss[..., :1] > -1
    return preds * mask

# def check_model_stability(model, val_data):
#     """
#     Check if model outputs are in reasonable range
#     """
#     model.eval()
#     with torch.no_grad():
#         xs_val, ys_val = val_data
#         model.set_initial_state(batch_size=len(xs_val))
#         state = model.get_state(detach=True)
#         outputs = model(xs_val, state, batch_first=True)[0]
        
#         # Check for problematic outputs
#         if torch.isnan(outputs).any() or torch.isinf(outputs).any():
#             return False
#         return True

class iMAMLTask:
    """
    Handles the train and validation loss for a single task with iMAML regularization
    Uses SINDy loss as the outer objective instead of validation loss
    """
    def __init__(self, model: BaseRNN, reg_param, train_data, dataset_val, sindy_config, loss_fn):
        device = model.device
        
        # Create stateless version of model for higher-order gradients
        self.fmodel = higher.monkeypatch(model, device=device, copy_initial_weights=True)
        
        self.train_xs, self.train_ys = train_data
        self.dataset_val = dataset_val
        self.sindy_config = sindy_config
        self.loss_fn = loss_fn
        self.model = model
        
        # Single regularization parameter for all parameters
        self.reg_param = reg_param
        
        self.sindy_loss_value = None
        self.train_loss_value = None
        
    def meta_reg_f(self, meta_params, task_params):
        """
        L2 biased regularization between meta-parameters (bias) and task parameters
        """
        reg_loss = 0.0
        reg_strength = self.reg_param
        
        for meta_p, task_p in zip(meta_params, task_params):
            if meta_p.requires_grad and task_p.requires_grad:
                reg_loss += reg_strength * ((meta_p - task_p) ** 2).sum()
        
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
    
    def sindy_loss_f(self, task_params, outer_params=None):
        """
        Compute differentiable SINDy loss for meta-optimization.
        """
        # Forward pass to get model outputs for validation data
        self.model.set_initial_state(batch_size=len(self.dataset_val.xs))
        state = self.model.get_state(detach=True)
        outputs = self.fmodel(self.dataset_val.xs, state, batch_first=True, params=task_params)[0]
        # outputs: [batch, seq, features] or [batch, features]

        # Prepare variables and controls for SINDy
        variables = [outputs]  # List of torch.Tensor
        control = [self.dataset_val.xs]  # List of torch.Tensor

        # Call differentiable SINDy fitting
        _, sindy_loss = fit_sindy_torch(
            variables=variables,
            control=control,
            rnn_modules=self.sindy_config["rnn_modules"],
            control_signals=self.sindy_config.get("control_parameters", []),
            library_setup=self.sindy_config["library_setup"],
            filter_setup=self.sindy_config["filter_setup"],
            polynomial_degree=self.sindy_config.get("polynomial_degree", 2),
            optimizer_alpha=self.sindy_config.get("optimizer_alpha", 1e-2),
            optimizer_nu=self.sindy_config.get("optimizer_nu", 1e-2),
            max_iter=self.sindy_config.get("max_iter", 100),
            verbose=False,
        )

        self.sindy_loss_value = sindy_loss.item()
        return sindy_loss



def fit_with_metaopt_sindy(
    model: BaseRNN,
    dataset_train: DatasetRNN,
    dataset_val: DatasetRNN,

    # SINDy config
    sindy_config: dict,

    model_optimizer: torch.optim.Optimizer = None,
    convergence_threshold: float = 1e-7,
    epochs: int = 1,
    batch_size: int = -1,
    bagging: bool = False,
    scheduler: bool = False,
    n_steps: int = -1,
    verbose: bool = True,
    path_save_checkpoints: str = None,
    
    # Meta-optimization parameters
    meta_update_interval=50,
    inner_steps=3,
    hypergradient_steps=3,
    outer_lr=1e-2,
    initial_reg_param=1e-4,

    ):

    """
    Fit RNN with iMAML meta-optimization using SINDy loss as outer objective
    Uses validation dataset for SINDy evaluation
    """
    
    if dataset_val is None:
        raise ValueError("Validation dataset is required for SINDy meta-optimization.")
    
    # Set batch size
    if batch_size == -1:
        batch_size = len(dataset_train)

    # Set bagging
    if bagging:
        batch_size = max(batch_size, 64)
        sampler = RandomSampler(dataset_train, replacement=True, num_samples=batch_size)
    else:
        raise NotImplementedError("Bagging=False is not implemented in this function.")

    # Leave at 0 for compatibility
    num_workers = 0
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, num_workers=num_workers)

    # Set up learning rate scheduler
    warmup_steps = 1024 if epochs > 1024 else 1
    if scheduler and model_optimizer is not None:
        default_lr = model_optimizer.param_groups[0]['lr'] + 0

        def warmup_lr_lambda(current_step):
            scale = 1e-1 / default_lr
            if current_step < warmup_steps * 0.8:
                return 0.1 * scale
            elif current_step < warmup_steps:
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

    # Meta-optimization setup with single regularization parameter
    reg_param = torch.tensor(initial_reg_param, device=model.device, requires_grad=True)
    
    # Optimizer for regularization parameter
    reg_optimizer = torch.optim.Adam([reg_param], lr=outer_lr)

    # Initialize losses and histories for logging/plotting
    train_loss = torch.tensor(0.)
    sindy_loss = torch.tensor(0.)

    history_train_loss = []
    history_sindy_loss = []
    history_hparams = []

    # Set up saving
    save_at_epoch = warmup_steps

    # Inner optimizer setup use original model lr
    inner_opt_class = hg.GradientDescent
    inner_lr = model_optimizer.param_groups[0]['lr'] if model_optimizer is not None else 1e-2
    inner_opt_kwargs = {'step_size': inner_lr}

    def get_inner_opt(train_loss_f):
        return inner_opt_class(train_loss_f, **inner_opt_kwargs)


    # Training loop
    try:
        for epoch in range(epochs):
            t_epoch_start = time.time()
            train_iter = iter(dataloader_train)

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
            # Handle NaNs/Infs in outputs for numerical stability
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)
            train_loss = loss_fn(outputs.reshape(-1, model._n_actions),
                                torch.argmax(ys.reshape(-1, model._n_actions), dim=1))
            
            train_loss.backward()
            # Gradient clipping for model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.2)
            model_optimizer.step()
            model_optimizer.zero_grad()

            # Meta-update every meta_update_interval epoch
            if (epoch + 1) % meta_update_interval == 0:
                # Check model stability first
                val_xs, val_ys = dataset_val.xs, dataset_val.ys
                if len(val_xs.shape) > 2:
                    val_xs = val_xs[0]  # Take first session if multiple sessions
                    val_ys = val_ys[0]

                # if check_model_stability(model, (val_xs, val_ys)):
                if True:  # Skip stability check for now

                    # Create iMAML task with SINDy objective
                    task = iMAMLTask(
                        model, 
                        reg_param, 
                        (xs, ys), 
                        dataset_val, 
                        sindy_config=sindy_config,
                        loss_fn=loss_fn
                    )
                    
                    # Inner optimization using stateless model
                    inner_opt = get_inner_opt(task.train_loss_f)
                    
                    # Initialize task-specific parameters
                    task_params = [p.detach().clone().requires_grad_(True) 
                                for p in model.parameters() if p.requires_grad]
                    
                    # Run inner loop
                    params_history = [task_params]
                    for inner_step in range(inner_steps):
                        new_params = inner_opt(params_history[-1], list(model.parameters()))
                        # Gradient clipping for inner loop parameters
                        torch.nn.utils.clip_grad_norm_(new_params, max_norm=0.05)
                        params_history.append(new_params)
                    
                    # Only proceed if inner optimization worked
                    if len(params_history) > 1:
                        final_task_params = params_history[-1]
                        
                        # Clear gradients before hypergradient computation
                        reg_optimizer.zero_grad()
                        
                        try:          
                            # Create a combined list of outer parameters including reg_param
                            outer_params = list(model.parameters()) + [reg_param]
                            
                            # Fixed point map for CG
                            cg_fp_map = hg.GradientDescent(loss_f=task.train_loss_f, step_size=inner_lr)
                            
                            # Solve using CG with SINDy loss as outer objective
                            hg_result = hg.CG(
                                final_task_params, 
                                outer_params,
                                K=hypergradient_steps, 
                                fp_map=cg_fp_map, 
                                outer_loss=task.sindy_loss_f  # Use SINDy loss instead of val loss
                            )
                            
                            # Check if reg_param got gradients from CG
                            if reg_param.grad is not None:
                                if torch.isnan(reg_param.grad).any():
                                    print("WARNING: NaN gradients detected for reg_param. Skipping meta-update this epoch.")
                                else:
                                    # Clipping regularization parameter gradients
                                    torch.nn.utils.clip_grad_norm_([reg_param], max_norm=0.05)
                                    reg_optimizer.step()
                            else:
                                print("WARNING: reg_param did not receive gradients from CG. Skipping meta-update this epoch.")
                            
                            # Clamp regularization parameter to prevent negative/extreme values
                            with torch.no_grad():
                                reg_param.clamp_(min=1e-4, max=1.0)

                            # Get SINDy loss for logging
                            sindy_loss = torch.tensor(task.sindy_loss_value if task.sindy_loss_value is not None else float('inf'))

                            # Zero out model parameter gradients after metaoptimization
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.zero_()
                        except Exception as e:
                            print(f"ERROR: CG computation failed at epoch {epoch + 1}: {e}")
                            print("Skipping meta-update this epoch due to CG failure.")
                            sindy_loss = torch.tensor(float('inf'))
                else:
                    print(f"WARNING: Model unstable at epoch {epoch + 1}. Skipping meta-update this epoch.")
                    sindy_loss = torch.tensor(float('inf'))

            # Update histories
            history_train_loss.append(train_loss.item())
            history_sindy_loss.append(sindy_loss.item())
            history_hparams.append(reg_param.detach().cpu().item())

            # Convergence check (based on SINDy loss)
            dloss = last_loss - sindy_loss
            convergence_value += recency_factor * (torch.abs(dloss).item() - convergence_value)
            converged = convergence_value < convergence_threshold
            last_loss = sindy_loss 

            # Logging
            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} --- L(Train): {train_loss:.4f}"
                msg += f"; L(SINDy): {sindy_loss:.4f}"
                msg += f"; Reg Param: {reg_param.item():.4f}"
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
                
            # Save checkpoint - ONLY model state_dict, NO reg_param
            if path_save_checkpoints and (epoch + 1) == save_at_epoch:
                torch.save(model.state_dict(), path_save_checkpoints.replace(".", f"_ep{epoch + 1}."))
                save_at_epoch *= 2
            
    except KeyboardInterrupt:
        print("Training interrupted. Continuing with further operations...")
        histories = [history_train_loss, history_sindy_loss, history_hparams]
        return model, model_optimizer, histories

    histories = [history_train_loss, history_sindy_loss, history_hparams]
    return model, model_optimizer, histories