# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
TEXGI Training Utilities

This module provides training-specific utilities for the TEXGI method:
- Temporal smoothness penalties for TEXGI attributions
- Generalized Pareto Distribution (GPD) sampling for adversarial baselines
- Expert prior penalties based on TEXGI attributions
- Feature importance aggregation from TEXGI outputs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any, Sequence


def sample_extreme_code(batch_size: int,
                        extreme_dim: int = 1,
                        device: str = "cpu",
                        xi: float = 0.3,
                        beta: float = 1.0) -> torch.Tensor:
    """
    Sample from Generalized Pareto Distribution (GPD) for adversarial baseline generation.

    Uses inverse CDF for u~Uniform(0,1):
        GPD(u) = beta/xi * ( (1 - u)^(-xi) - 1 )

    Parameters
    ----------
    batch_size : int
        Number of samples to generate
    extreme_dim : int, default 1
        Dimension of the extreme code
    device : str, default "cpu"
        Device for tensor allocation
    xi : float, default 0.3
        Shape parameter of GPD
    beta : float, default 1.0
        Scale parameter of GPD

    Returns
    -------
    extreme_codes : torch.Tensor of shape (batch_size, extreme_dim)
        Samples from GPD
    """
    u = torch.rand(batch_size, extreme_dim, device=device).clamp(1e-6, 1 - 1e-6)
    e = beta / xi * ((1 - u) ** (-xi) - 1.0)
    return e


def attribution_temporal_l1(phi_tbd: torch.Tensor) -> torch.Tensor:
    """
    L1 temporal smoothness penalty on TEXGI attributions.

    Implements Eq.(smooth) from the paper:
        sum_{i,b,d} |phi_{t+1,b,d} - phi_{t,b,d}|

    Parameters
    ----------
    phi_tbd : torch.Tensor of shape (T, B, D)
        TEXGI attributions with T time bins, B batch samples, D features

    Returns
    -------
    penalty : torch.Tensor (scalar)
        Raw (unweighted) temporal smoothness penalty

    Raises
    ------
    ValueError
        If phi tensor does not have 3 dimensions [T, B, D]
    """
    if phi_tbd.dim() != 3:
        raise ValueError("phi tensor must have shape [T, B, D] for temporal smoothing")

    if phi_tbd.size(0) <= 1:
        return phi_tbd.new_zeros(())

    diff = phi_tbd[1:] - phi_tbd[:-1]
    return diff.abs().sum()


def aggregate_importance(phi_tbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate TEXGI attributions to compute feature importance.

    Parameters
    ----------
    phi_tbd : torch.Tensor of shape (T, B, D)
        TEXGI attributions with T time bins, B batch samples, D features

    Returns
    -------
    importance_abs : torch.Tensor of shape (D,)
        Per-feature global importance = mean_{t,b} |phi|
    importance_dir : torch.Tensor of shape (D,)
        Per-feature directional mean = mean_{t,b} phi
    """
    importance_abs = phi_tbd.abs().mean(dim=(0, 1))  # [D]
    importance_dir = phi_tbd.mean(dim=(0, 1))  # [D]
    return importance_abs, importance_dir


def expert_penalty(phi_tbd: torch.Tensor,
                   important_idx: Sequence[int]) -> torch.Tensor:
    """
    Expert prior penalty for TEXGI attributions.

    Implements expert knowledge constraints by penalizing:
    1. Important features that have below-average importance
    2. Non-important features (encourage sparsity)

    Parameters
    ----------
    phi_tbd : torch.Tensor of shape (T, B, D)
        TEXGI attributions
    important_idx : sequence of int
        Indices of features marked as important by domain experts

    Returns
    -------
    penalty : torch.Tensor (scalar)
        Expert prior penalty value

    Raises
    ------
    ValueError
        If phi tensor does not have 3 dimensions [T, B, D]
    """
    if phi_tbd.dim() != 3:
        raise ValueError("phi tensor must have shape [T, B, D] for expert penalty")

    device = phi_tbd.device

    # Compute L2 norms across all time bins and batch elements
    norms = torch.sqrt(phi_tbd.pow(2).sum(dim=(0, 1)) + 1e-12)  # [D]

    if norms.numel() == 0:
        return torch.zeros((), device=device)

    bar_s = norms.mean()  # Average norm
    total = torch.zeros((), device=device)

    # Create mask for important features
    important_mask = torch.zeros_like(norms, dtype=torch.bool)

    if important_idx:
        important_tensor = torch.tensor(important_idx, device=device, dtype=torch.long)
        important_mask[important_tensor] = True
        # Penalize important features that have below-average importance
        total = total + F.relu(bar_s - norms[important_tensor]).sum()

    # Penalize non-important features (encourage sparsity)
    non_important = ~important_mask
    if non_important.any():
        total = total + norms[non_important].sum()

    return total


def resolve_important_feature_indices(
    expert_config: Optional[Dict[str, Any]],
    feat2idx: Dict[str, int]
) -> List[int]:
    """
    Extract important feature indices from expert configuration.

    Parameters
    ----------
    expert_config : dict, optional
        Expert configuration containing feature rules
    feat2idx : dict
        Mapping from feature names to indices

    Returns
    -------
    indices : list of int
        Sorted list of important feature indices
    """
    if not expert_config:
        return []

    names: List[str] = []

    # Extract from various configuration formats
    for key in ("important_features", "important", "I"):
        val = expert_config.get(key)
        if isinstance(val, (list, tuple, set)):
            names.extend(str(v) for v in val)

    # Handle rule-based specifications
    rules = expert_config.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            fname = rule.get("feature")
            if not fname:
                continue
            relation = str(rule.get("relation", "")).lower()
            if relation in {">=mean", ">=avg", "important"} or bool(rule.get("important", False)):
                names.append(str(fname))

    # Convert to indices
    idx = {feat2idx[name] for name in names if name in feat2idx}
    return sorted(idx)


def masked_bce_nll(hazards: torch.Tensor,
                   labels: torch.Tensor,
                   masks: torch.Tensor) -> torch.Tensor:
    """
    Masked binary cross-entropy negative log-likelihood for survival hazards.

    Parameters
    ----------
    hazards : torch.Tensor of shape (B, T)
        Predicted hazard probabilities in (0, 1)
    labels : torch.Tensor of shape (B, T)
        Binary event indicators in {0, 1}
    masks : torch.Tensor of shape (B, T)
        Binary masks indicating valid timesteps (1=valid, 0=masked)

    Returns
    -------
    nll : torch.Tensor (scalar)
        Masked BCE loss value
    """
    eps = 1e-6
    hazards = hazards.clamp(eps, 1 - eps)
    bce = -(labels * torch.log(hazards) + (1.0 - labels) * torch.log(1.0 - hazards))
    masked = bce * masks
    denom = masks.sum().clamp_min(1.0)
    return masked.sum() / denom


def topk_feature_importance(phi_tbd: torch.Tensor,
                           feature_names: List[str],
                           k: int = 10) -> pd.DataFrame:
    """
    Get top-k most important features from TEXGI attributions.

    Parameters
    ----------
    phi_tbd : torch.Tensor of shape (T, B, D)
        TEXGI attributions
    feature_names : list of str
        Names of features
    k : int, default 10
        Number of top features to return

    Returns
    -------
    importance_df : pd.DataFrame
        DataFrame with columns ['feature', 'importance'] sorted by importance
    """
    importance_abs, _ = aggregate_importance(phi_tbd)
    importance = importance_abs.detach().cpu().numpy()

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    return df.reset_index(drop=True).head(k)


def integrated_gradients_time(
    model: nn.Module,
    X: torch.Tensor,
    X_baseline: torch.Tensor,
    hazard_index: int,
    M: int = 20,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute Integrated Gradients for a specific time bin.

    Computes IG along the straight path from baseline to input for
    a single hazard time index.

    Parameters
    ----------
    model : nn.Module
        Model that outputs hazards of shape (B, T)
    X : torch.Tensor of shape (B, D)
        Input features in real space
    X_baseline : torch.Tensor of shape (B, D)
        Baseline features in real space
    hazard_index : int
        Target time bin index
    M : int, default 20
        Number of integration steps
    forward_kwargs : dict, optional
        Additional keyword arguments for model forward pass

    Returns
    -------
    attributions : torch.Tensor of shape (B, D)
        IG attributions for the specified time bin
    """
    assert X.shape == X_baseline.shape
    device = X.device

    alphas = torch.linspace(0.0, 1.0, steps=M + 1, device=device)[1:]  # exclude 0
    X_diff = (X - X_baseline)

    grad_terms: List[torch.Tensor] = []
    kwargs = forward_kwargs or {}

    for alpha in alphas:
        X_path = X_baseline + alpha * X_diff
        X_path = X_path.detach().clone().requires_grad_(True)

        hazards = model(X_path, **kwargs) if kwargs else model(X_path)
        output = hazards[:, hazard_index]

        grads = torch.autograd.grad(
            output.sum(),
            X_path,
            retain_graph=False,
            create_graph=False,
        )[0]
        grad_terms.append(grads)

    attributions = torch.stack(grad_terms, dim=0).mean(dim=0) * X_diff
    return attributions


def texgi_time_series(
    model: nn.Module,
    X: torch.Tensor,
    X_baseline: Optional[torch.Tensor] = None,
    M: int = 20,
    t_sample: Optional[int] = None,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute TEXGI attributions per time-bin.

    Parameters
    ----------
    model : nn.Module
        Model that outputs hazards of shape (B, T)
    X : torch.Tensor of shape (B, D)
        Input features in real space
    X_baseline : torch.Tensor, optional
        Baseline features. If None, uses quantile-based baseline.
    M : int, default 20
        Number of integration steps
    t_sample : int, optional
        Number of time bins to sample. If None, uses all bins.
    forward_kwargs : dict, optional
        Additional keyword arguments for model forward pass

    Returns
    -------
    phi : torch.Tensor of shape (T', B, D)
        TEXGI attributions per time bin. T' may be less than T if t_sample is set.
    """
    if not torch.is_tensor(X):
        raise TypeError(f"X must be a torch.Tensor, received {type(X).__name__}")

    device = X.device

    # Create baseline if not provided
    if X_baseline is None:
        q_hi = torch.quantile(X, 0.98, dim=0, keepdim=True)
        X_baseline = q_hi.repeat(X.size(0), 1)

    # Determine number of time bins
    with torch.no_grad():
        if forward_kwargs:
            T = model(X[:1], **forward_kwargs).shape[1]
        else:
            T = model(X[:1]).shape[1]

    # Select time bin indices
    if t_sample is not None and 0 < t_sample < T:
        step = max(1, math.floor(T / t_sample))
        t_indices = list(range(0, T, step))[:t_sample]
    else:
        t_indices = list(range(T))

    # Compute IG for each time bin
    phi_list = []
    for t in t_indices:
        ig_t = integrated_gradients_time(
            model,
            X,
            X_baseline,
            hazard_index=t,
            M=M,
            forward_kwargs=forward_kwargs,
        )
        phi_list.append(ig_t)

    phi = torch.stack(phi_list, dim=0)
    return phi


def standardize_features(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standardize features to zero mean and unit variance.

    Parameters
    ----------
    X : torch.Tensor of shape (N, D)
        Input features

    Returns
    -------
    X_standardized : torch.Tensor of shape (N, D)
        Standardized features
    mu : torch.Tensor of shape (1, D)
        Feature means
    std : torch.Tensor of shape (1, D)
        Feature standard deviations
    """
    with torch.no_grad():
        mu = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_standardized = (X - mu) / std
    return X_standardized, mu, std


def destandardize_features(X_std: torch.Tensor,
                          mu: torch.Tensor,
                          std: torch.Tensor) -> torch.Tensor:
    """
    Convert standardized features back to original scale.

    Parameters
    ----------
    X_std : torch.Tensor of shape (N, D)
        Standardized features
    mu : torch.Tensor of shape (1, D)
        Feature means
    std : torch.Tensor of shape (1, D)
        Feature standard deviations

    Returns
    -------
    X : torch.Tensor of shape (N, D)
        Features in original scale
    """
    return X_std * std + mu
