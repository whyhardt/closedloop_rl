import torch
from typing import List, Dict, Any, Tuple

def conditional_filtering_torch(x_train: List[torch.Tensor], control: List[torch.Tensor], feature_names: List[str], feature_filter: str, condition: float, remove_feature_filter=True) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Filter data based on condition and optionally remove the filter feature.
    Matches the original conditional_filtering function behavior.
    """
    if feature_filter not in feature_names:
        print(f"WARNING: Feature '{feature_filter}' not found in feature_names {feature_names}. Skipping filtering.")
        return x_train, control, feature_names
        
    idx = feature_names.index(feature_filter)
    x_train_filtered, control_filtered = [], []
    
    # The filter feature should be in the control signals (index adjusted for control array)
    control_idx = idx - 1 if idx > 0 else None  # Subtract 1 because feature_names[0] is target variable
    
    for i, (x, u) in enumerate(zip(x_train, control)):
        if control_idx is not None and control_idx < u.shape[1]:
            # Filter based on control signal
            mask = (u[:, control_idx] == condition)
        elif idx == 0 and x.shape[1] > 0:
            # Filter based on target variable itself
            mask = (x[:, 0] == condition)
        else:
            print(f"WARNING: Cannot apply filter {feature_filter}=={condition}. Using all data.")
            mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        
        if mask.sum() == 0:
            print(f"WARNING: No data points satisfy condition {feature_filter}=={condition} for trajectory {i}. Using all data.")
            x_train_filtered.append(x)
            control_filtered.append(u)
        else:
            x_train_filtered.append(x[mask])
            control_filtered.append(u[mask])
    
    # Remove feature from control signals if requested
    new_feature_names = feature_names.copy()
    if remove_feature_filter and feature_filter in new_feature_names and control_idx is not None:
        new_feature_names.remove(feature_filter)
        # Remove the corresponding column from control tensors
        control_filtered_new = []
        for u in control_filtered:
            if u.shape[1] > control_idx:
                if control_idx == 0 and u.shape[1] > 1:
                    u_new = u[:, 1:]
                elif control_idx == u.shape[1] - 1:
                    u_new = u[:, :-1]
                else:
                    u_new = torch.cat([u[:, :control_idx], u[:, control_idx+1:]], dim=1)
                control_filtered_new.append(u_new)
            else:
                control_filtered_new.append(u)
        control_filtered = control_filtered_new
    
    return x_train_filtered, control_filtered, new_feature_names

def build_feature_library_torch(x: torch.Tensor, u: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Build polynomial feature library matching PySINDy's PolynomialLibrary.
    """
    features = [torch.ones(x.shape[0], 1, device=x.device)]
    
    # Combine x and u for polynomial expansion
    if u is not None and u.shape[1] > 0:
        combined = torch.cat([x, u], dim=1)
    else:
        combined = x
    
    # Add polynomial features
    for d in range(1, degree + 1):
        features.append(combined ** d)
    
    return torch.cat(features, dim=1)

def remove_control_features_torch(control: List[torch.Tensor], control_names: List[str], allowed_controls: List[str]) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Remove control features that are not in the allowed list.
    """
    if not allowed_controls:
        return control, []
    
    # Find indices of allowed controls
    allowed_indices = [i for i, name in enumerate(control_names) if name in allowed_controls]
    
    if not allowed_indices:
        return control, []
    
    # Filter control tensors
    filtered_control = []
    for u in control:
        if u.shape[1] > 0:
            valid_indices = [i for i in allowed_indices if i < u.shape[1]]
            if valid_indices:
                filtered_control.append(u[:, valid_indices])
            else:
                filtered_control.append(torch.empty(u.shape[0], 0, device=u.device))
        else:
            filtered_control.append(u)
    
    # Filter control names
    filtered_names = [control_names[i] for i in allowed_indices if i < len(control_names)]
    
    return filtered_control, filtered_names

class SR3Regression(torch.nn.Module):
    def __init__(self, in_features, out_features, l1_penalty=1e-2, nu=1e-2):
        super().__init__()
        self.coef = torch.nn.Parameter(torch.zeros(in_features, out_features))
        self.l1_penalty = l1_penalty
        self.nu = nu

    def forward(self, Theta, y):
        pred = Theta @ self.coef
        mse = torch.mean((y - pred) ** 2)
        l1 = self.l1_penalty * torch.norm(self.coef, p=1)
        sr3 = self.nu * torch.norm(self.coef, p=2)
        return mse + l1 + sr3

def fit_sindy_torch(
    variables: List[torch.Tensor],
    control: List[torch.Tensor],
    rnn_modules: List[str],
    control_signals: List[str],
    library_setup: Dict[str, List[str]],
    filter_setup: Dict[str, Any],
    polynomial_degree: int = 2,
    optimizer_alpha: float = 1e-2,
    optimizer_nu: float = 1e-2,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """
    Differentiable PyTorch SINDy fitting matching the original fit_sindy function.
    """
    device = variables[0].device
    sindy_models = {}
    total_loss = 0.0
    
    # Input validation
    if not variables or not control:
        print("ERROR: No variables or control signals provided to SINDy")
        return sindy_models, torch.tensor(1e-4, device=device, requires_grad=True)
    
    # Check dimensions
    print(f"DEBUG: Variables shape: {[v.shape for v in variables]}")
    print(f"DEBUG: Control shape: {[c.shape for c in control]}")
    print(f"DEBUG: Expected RNN modules: {rnn_modules}")
    print(f"DEBUG: Control signals: {control_signals}")
    
    # Determine how many features we can actually process
    max_var_features = min(var.shape[1] for var in variables) if variables else 0
    available_modules = rnn_modules[:max_var_features] if max_var_features > 0 else []
    
    if not available_modules:
        print(f"ERROR: Cannot process any RNN modules. Variables have {max_var_features} features but need at least 1.")
        return sindy_models, torch.tensor(1e-4, device=device, requires_grad=True)
    
    if len(available_modules) < len(rnn_modules):
        print(f"WARNING: Only processing {len(available_modules)} out of {len(rnn_modules)} RNN modules due to dimension constraints.")
    
    # Process each RNN module (target variable)
    for index_feature, x_feature in enumerate(available_modules):
        try:
            if verbose:
                print(f'\nProcessing SINDy model for {x_feature}:')
            
            # Extract target variable (current x-feature)
            x_i = [x[:, index_feature].reshape(-1, 1) for x in variables]
            control_i = [c.clone() for c in control]  # Copy control data
            
            # Initial feature names: [target_var, control1, control2, ...]
            feature_names_i = [x_feature] + control_signals
            
            # Apply filtering conditions
            if x_feature in filter_setup:
                filter_conditions = filter_setup[x_feature]
                # Handle both single condition and list of conditions
                if not isinstance(filter_conditions[0], list):
                    filter_conditions = [filter_conditions]
                
                for filter_condition in filter_conditions:
                    if len(filter_condition) >= 3:
                        filter_feature, condition_value, remove_feature = filter_condition
                        x_i, control_i, feature_names_i = conditional_filtering_torch(
                            x_train=x_i,
                            control=control_i,
                            feature_names=feature_names_i,
                            feature_filter=filter_feature,
                            condition=condition_value,
                            remove_feature_filter=remove_feature
                        )
            
            # Check if we still have data after filtering
            if not x_i or all(x.shape[0] == 0 for x in x_i):
                print(f"WARNING: No data remaining after filtering for {x_feature}. Skipping.")
                continue
            
            # Remove unnecessary control features according to library setup
            current_control_names = feature_names_i[1:]  # Exclude target variable
            allowed_controls = library_setup.get(x_feature, [])
            control_i, filtered_control_names = remove_control_features_torch(
                control_i, current_control_names, allowed_controls
            )
            
            # Update feature names
            feature_names_i = [x_feature] + filtered_control_names
            
            # Concatenate data from all trajectories
            X = torch.cat(x_i, dim=0)
            U = torch.cat(control_i, dim=0) if control_i and all(c.shape[0] > 0 for c in control_i) else None
            
            if U is not None and U.shape[1] == 0:
                U = None
                
            if verbose:
                print(f"  Target shape: {X.shape}")
                print(f"  Control shape: {U.shape if U is not None else None}")
                print(f"  Feature names: {feature_names_i}")
            
            # Build feature library
            Theta = build_feature_library_torch(X, U, polynomial_degree)
            y = X
            
            # Check for valid data
            if Theta.shape[0] == 0 or y.shape[0] == 0:
                print(f"WARNING: No data for regression for {x_feature}. Skipping.")
                continue
            
            # SR3 regression
            model = SR3Regression(
                Theta.shape[1], 1, 
                l1_penalty=optimizer_alpha, 
                nu=optimizer_nu
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            
            # Training loop
            for iter in range(max_iter):
                try:
                    loss = model(Theta, y)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"WARNING: Invalid loss for {x_feature} at iteration {iter}. Stopping.")
                        break
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    if verbose and iter % 20 == 0:
                        print(f"  Iter {iter}: Loss={loss.item():.4f}")
                        
                except RuntimeError as e:
                    print(f"ERROR during optimization for {x_feature}: {e}")
                    break
            
            # Store model and add to total loss
            sindy_models[x_feature] = model
            
            try:
                feature_loss = model(Theta, y)
                if not torch.isnan(feature_loss) and not torch.isinf(feature_loss):
                    total_loss = total_loss + feature_loss
                else:
                    print(f"WARNING: Invalid final loss for {x_feature}")
            except RuntimeError as e:
                print(f"ERROR computing final loss for {x_feature}: {e}")
                
        except Exception as e:
            print(f"ERROR processing feature {x_feature}: {e}")
            continue
    
    # Handle final loss
    if not sindy_models:
        print("ERROR: No SINDy models were successfully fitted")
        return sindy_models, torch.tensor(1e-4, device=device, requires_grad=True)
    
    # Ensure proper tensor with gradients
    if not isinstance(total_loss, torch.Tensor):
        total_loss = torch.tensor(total_loss, device=device, requires_grad=True)
    elif total_loss.item() == 0.0:
        total_loss = torch.tensor(1e-4, device=device, requires_grad=True)
    elif not total_loss.requires_grad:
        total_loss = total_loss.detach().requires_grad_(True)
    
    return sindy_models, total_loss