# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Advanced TEXGISA implementation with enhancements from SAWEB.

This module contains advanced functions ported from SAWEB including:
- Time-dependent Integrated Gradients (per-time-bin computation)
- Generalized Pareto Distribution sampling for extreme codes
- Advanced adversarial generator training with proximity constraints
- L2 norm-based expert penalty implementation
- Attribution temporal smoothness regularization
- MySATrainer class for better training abstraction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Sequence
from torch.utils.data import DataLoader


# ---------------------- Advanced Generator with GPD Sampling ----------------------

class TabularGeneratorWithRealInput(nn.Module):
    """
    Advanced generator G(x, z, e) -> x_adv
    - x: real standardized features [B, D]
    - z: latent noise [B, Z]
    - e: extreme code sampled from Generalized Pareto dist [B, E]

    The generator outputs a baseline lying near the data manifold but
    shifted toward an extreme direction encoded by e.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16, extreme_dim: int = 1,
                 hidden: int = 256, depth: int = 3):
        super().__init__()
        D = input_dim + latent_dim + extreme_dim
        layers = []
        h = hidden
        layers.append(nn.Linear(D, h))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(h, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z, e], dim=1))


def sample_extreme_code(batch_size: int, extreme_dim: int = 1, device: str = "cpu",
                        xi: float = 0.3, beta: float = 1.0) -> torch.Tensor:
    """
    Sample from Generalized Pareto Distribution (GPD) using inverse CDF.

    For u~Uniform(0,1):
      GPD(u) = beta/xi * ((1 - u)^(-xi) - 1)

    Maps to (0, +∞). For stability, we cap u away from 1.

    Parameters
    ----------
    batch_size : int
        Number of samples
    extreme_dim : int
        Dimension of extreme code
    device : str
        Device for computation
    xi : float
        Shape parameter
    beta : float
        Scale parameter

    Returns
    -------
    torch.Tensor
        Extreme codes [batch_size, extreme_dim]
    """
    u = torch.rand(batch_size, extreme_dim, device=device).clamp(1e-6, 1 - 1e-6)
    e = beta / xi * ((1 - u) ** (-xi) - 1.0)
    return e


# ---------------------- Standardization Utilities ----------------------

@torch.no_grad()
def _standardize_fit(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and std for standardization."""
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, std


def _standardize_apply(X: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Apply standardization."""
    return (X - mu) / std


def _destandardize(Xs: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Reverse standardization."""
    return Xs * std + mu


# ---------------------- Advanced Generator Training ----------------------

def train_adv_generator(
    model: nn.Module,
    X_tr: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 256,
    latent_dim: int = 16,
    extreme_dim: int = 1,
    lr: float = 1e-3,
    alpha_dist: float = 1.0,
    device: Optional[str] = None,
    modality_mask: Optional[torch.Tensor] = None,
    checkpoint_path: str = "texgisa_generator.pt",
) -> Tuple[TabularGeneratorWithRealInput, Dict[str, torch.Tensor]]:
    """
    Train a generator G to produce adversarial extreme baselines.

    Objective (max-risk with proximity):
       maximize  Risk(model(x_adv)) - alpha_dist * ||x_adv - x||_2^2
    where Risk = sum_t hazard_t (earlier event -> larger risk).

    Notes:
      - We optimize the negative of above because we use Adam on G to MINIMIZE.
      - x, x_adv are standardized here; caller will de-standardize when needed.

    Parameters
    ----------
    model : nn.Module
        Survival model
    X_tr : torch.Tensor
        Training features [N, D]
    mu, std : torch.Tensor
        Standardization parameters
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    latent_dim : int
        Latent dimension
    extreme_dim : int
        Extreme code dimension
    lr : float
        Learning rate
    alpha_dist : float
        Distance penalty weight
    device : str, optional
        Device for computation
    modality_mask : torch.Tensor, optional
        Modality availability mask
    checkpoint_path : str
        Path to save generator checkpoint

    Returns
    -------
    G : TabularGeneratorWithRealInput
        Trained generator
    ref_stats : dict
        Reference statistics (mu, std)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()  # freeze model during G training

    G = TabularGeneratorWithRealInput(X_tr.shape[1], latent_dim, extreme_dim).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=lr)

    N = X_tr.shape[0]
    idx = torch.randperm(N)
    X_std = _standardize_apply(X_tr.to(device), mu.to(device), std.to(device))
    if modality_mask is not None:
        modality_mask = modality_mask.to(device)

    steps_per_epoch = max(1, N // batch_size)

    best_loss = float('inf')

    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for i in range(steps_per_epoch):
            sl = i * batch_size
            sr = min(N, sl + batch_size)
            xb = X_std[idx[sl:sr]]  # standardized input

            z = torch.randn(xb.size(0), latent_dim, device=device)
            e = sample_extreme_code(xb.size(0), extreme_dim=extreme_dim, device=device, xi=0.3, beta=1.0)

            x_adv = G(xb, z, e)  # standardized baseline
            x_adv_real = _destandardize(x_adv, mu.to(device), std.to(device))  # back to real space

            # Risk proxy: sum of hazards
            mask_batch = None
            if modality_mask is not None:
                mask_batch = modality_mask[idx[sl:sr]]

            with torch.no_grad():
                if mask_batch is not None and hasattr(model, 'forward') and 'modality_mask' in model.forward.__code__.co_varnames:
                    hazards = model(x_adv_real, modality_mask=mask_batch)
                else:
                    hazards = model(x_adv_real)
                risk = hazards.sum(dim=1)  # larger -> earlier event

            # Proximity in standardized space
            dist = (x_adv - xb).pow(2).sum(dim=1)

            # We minimize: -risk + alpha * dist
            loss = (-risk + alpha_dist * dist).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt.step()

            ep_loss += loss.item()

        avg_loss = ep_loss / steps_per_epoch
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save checkpoint
            try:
                torch.save(G.state_dict(), checkpoint_path)
            except Exception:
                pass

        if ep % 50 == 0 or ep == 1:
            print(f"[Generator Training] Epoch {ep:03d}  Loss={avg_loss:.4f}")

    ref_stats = {"mu": mu.to(device), "std": std.to(device)}
    return G, ref_stats


# ---------------------- Time-dependent Integrated Gradients ----------------------

def integrated_gradients_time(
    f: nn.Module,
    X: torch.Tensor,           # [B, D] (real space, requires_grad can be True)
    X_baseline: torch.Tensor,  # [B, D] (real space)
    hazard_index: int,         # target time-bin
    M: int = 20,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute Integrated Gradients for one time index t along the straight path
    from baseline to input.

    Parameters
    ----------
    f : nn.Module
        Model to compute gradients for
    X : torch.Tensor
        Input features [B, D]
    X_baseline : torch.Tensor
        Baseline features [B, D]
    hazard_index : int
        Target time-bin index
    M : int
        Number of integration steps
    forward_kwargs : dict, optional
        Additional kwargs for model forward pass

    Returns
    -------
    torch.Tensor
        IG attributions [B, D] for this time index
    """
    assert X.shape == X_baseline.shape
    device = X.device
    alphas = torch.linspace(0.0, 1.0, steps=M + 1, device=device)[1:]  # exclude 0
    Xdiff = (X - X_baseline)

    atts_terms: List[torch.Tensor] = []
    kwargs = forward_kwargs or {}

    # Put model in eval mode to avoid batch norm issues
    training_mode = f.training
    f.eval()

    for a in alphas:
        Xpath = X_baseline + a * Xdiff
        Xpath.requires_grad_(True)
        hazards = f(Xpath, **kwargs) if kwargs else f(Xpath)  # [B, T]
        out = hazards[:, hazard_index]  # focus on bin t
        grads = torch.autograd.grad(
            out.sum(),
            Xpath,
            retain_graph=False,
            create_graph=False,
        )[0]
        atts_terms.append(grads)

    f.train(training_mode)  # Restore original mode

    atts = torch.stack(atts_terms, dim=0).mean(dim=0) * Xdiff
    return atts


def texgi_time_series(
    f: nn.Module,
    X: torch.Tensor,  # [B, D] real space
    G: Optional[TabularGeneratorWithRealInput],
    ref_stats: Dict[str, torch.Tensor],
    M: int = 20,
    latent_dim: int = 16,
    extreme_dim: int = 1,
    t_sample: Optional[int] = None,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute TEXGI per time-bin, returning phi with shape [T, B, D].

    Baseline is produced by adversarial generator conditioned on (x, z, e).

    Parameters
    ----------
    f : nn.Module
        Model to compute gradients for
    X : torch.Tensor
        Input features [B, D]
    G : TabularGeneratorWithRealInput, optional
        Trained generator for baselines
    ref_stats : dict
        Standardization parameters (mu, std)
    M : int
        Number of integration steps
    latent_dim : int
        Latent dimension for generator
    extreme_dim : int
        Extreme code dimension
    t_sample : int, optional
        Number of time bins to subsample (for efficiency)
    forward_kwargs : dict, optional
        Additional kwargs for model forward pass

    Returns
    -------
    torch.Tensor
        TEXGI attributions [T, B, D] or [T', B, D] if subsampled
    """
    if not torch.is_tensor(X):
        raise TypeError(f"texgi_time_series expects a torch.Tensor; received {type(X).__name__}.")

    device = X.device

    # Prepare baseline
    if G is not None:
        with torch.no_grad():
            mu, std = ref_stats["mu"], ref_stats["std"]
            Xstd = _standardize_apply(X, mu, std)
            z = torch.randn(X.size(0), latent_dim, device=device)
            e = sample_extreme_code(X.size(0), extreme_dim=extreme_dim, device=device)
            Xb_std = G(Xstd, z, e)
            X_baseline = _destandardize(Xb_std, mu, std)
    else:
        # Fallback: use high-quantile statistic
        q_hi = torch.quantile(X, 0.98, dim=0, keepdim=True)
        X_baseline = q_hi.repeat(X.size(0), 1)

    # Get number of time bins
    with torch.no_grad():
        # Put model in eval mode to avoid batch norm issues with single sample
        training_mode = f.training
        f.eval()
        if forward_kwargs:
            T = f(X[:1], **forward_kwargs).shape[1]
        else:
            T = f(X[:1]).shape[1]
        f.train(training_mode)  # Restore original mode

    # Select time bins to compute (subsample for efficiency if requested)
    if t_sample is not None and t_sample > 0 and t_sample < T:
        import math
        step = max(1, math.floor(T / t_sample))
        t_indices = list(range(0, T, step))[:t_sample]
    else:
        t_indices = list(range(T))

    phi_list = []
    for t in t_indices:
        ig_t = integrated_gradients_time(
            f,
            X,
            X_baseline,
            hazard_index=t,
            M=M,
            forward_kwargs=forward_kwargs,
        )
        phi_list.append(ig_t)

    # [T', B, D] where T' is number of sampled time bins
    phi = torch.stack(phi_list, dim=0)
    return phi


# ---------------------- Attribution Temporal Smoothness ----------------------

def attribution_temporal_l1(phi_tbd: torch.Tensor) -> torch.Tensor:
    """
    L1 temporal smoothness penalty on TEXGI attributions.

    Implements sum_{t,b,d} |phi_{t+1,b,d} - phi_{t,b,d}|.

    Parameters
    ----------
    phi_tbd : torch.Tensor
        Attributions tensor [T, B, D]

    Returns
    -------
    torch.Tensor
        Scalar temporal smoothness penalty
    """
    if phi_tbd.dim() != 3:
        raise ValueError("phi tensor must have shape [T, B, D] for temporal smoothing")
    if phi_tbd.size(0) <= 1:
        return phi_tbd.new_zeros(())
    diff = phi_tbd[1:] - phi_tbd[:-1]
    return diff.abs().sum()


# ---------------------- Expert Penalty with L2 Norm ----------------------

def _aggregate_importance(phi_tbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aggregate attributions across time and batch dimensions.

    Parameters
    ----------
    phi_tbd : torch.Tensor
        Attributions [T, B, D]

    Returns
    -------
    imp_abs : torch.Tensor
        Per-feature global importance = mean_{t,b} |phi| -> [D]
    imp_dir : torch.Tensor
        Per-feature directional mean = mean_{t,b} phi -> [D]
    """
    imp_abs = phi_tbd.abs().mean(dim=(0, 1))  # [D]
    imp_dir = phi_tbd.mean(dim=(0, 1))        # [D]
    return imp_abs, imp_dir


def _resolve_important_feature_indices(
    expert_config: Optional[Dict[str, Any]],
    feat2idx: Dict[str, int]
) -> List[int]:
    """
    Extract the set I (important features) from the expert configuration.

    Parameters
    ----------
    expert_config : dict, optional
        Expert configuration with important features
    feat2idx : dict
        Feature name to index mapping

    Returns
    -------
    list
        Sorted list of important feature indices
    """
    if not expert_config:
        return []

    names: List[str] = []

    # Check for explicit important features
    for key in ("important_features", "important", "I"):
        val = expert_config.get(key)
        if isinstance(val, (list, tuple, set)):
            names.extend(str(v) for v in val)

    # Backward compatibility: treat old rule entries with relation >=mean as important
    for rule in expert_config.get("rules", []) if isinstance(expert_config.get("rules"), list) else []:
        if not isinstance(rule, dict):
            continue
        fname = rule.get("feature")
        if not fname:
            continue
        relation = str(rule.get("relation", "")).lower()
        if relation in {">=mean", ">=avg", "important"} or bool(rule.get("important", False)):
            names.append(str(fname))

    idx = {feat2idx[name] for name in names if name in feat2idx}
    return sorted(idx)


def expert_penalty(phi_tbd: torch.Tensor, important_idx: Sequence[int]) -> torch.Tensor:
    """
    Expert prior penalty Ω_expert(Φ) following Equation 18 from the paper.

    For important features I:
        Penalty = ReLU(mean_norm - ||Φ_i||_2) if i in I
    For non-important features:
        Penalty = ||Φ_i||_2

    Parameters
    ----------
    phi_tbd : torch.Tensor
        Attributions [T, B, D]
    important_idx : sequence
        Indices of important features

    Returns
    -------
    torch.Tensor
        Scalar expert penalty
    """
    if phi_tbd.dim() != 3:
        raise ValueError("phi tensor must have shape [T, B, D] for expert penalty")

    device = phi_tbd.device
    # ||Φ_l||_2 across all time bins and batch elements
    norms = torch.sqrt(phi_tbd.pow(2).sum(dim=(0, 1)) + 1e-12)  # [D]
    if norms.numel() == 0:
        return torch.zeros((), device=device)

    bar_s = norms.mean()
    total = torch.zeros((), device=device)

    important_mask = torch.zeros_like(norms, dtype=torch.bool)
    if important_idx:
        important_tensor = torch.tensor(important_idx, device=device, dtype=torch.long)
        important_mask[important_tensor] = True
        total = total + F.relu(bar_s - norms[important_tensor]).sum()

    non_important = ~important_mask
    if non_important.any():
        total = total + norms[non_important].sum()

    return total


# ---------------------- MySATrainer Class ----------------------

class MySATrainer:
    """
    Advanced trainer for TEXGISA model with better abstraction and efficiency.

    Features:
    - Lazy generator initialization
    - Batch/time subsampling for efficient IG computation
    - Flexible input handling (tensor/dict/modality support)
    - Checkpoint management
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: Optional[str] = None,
        lambda_smooth: float = 0.0,
        lambda_expert: float = 0.0,
        expert_rules: Optional[Dict[str, Any]] = None,
        feat2idx: Optional[Dict[str, int]] = None,
        # TEXGI settings
        ig_steps: int = 20,
        latent_dim: int = 16,
        extreme_dim: int = 1,
        ig_batch_samples: int = 64,  # Subsample batch for IG computation
        ig_time_subsample: Optional[int] = None,  # Subsample time bins
        # Generator settings
        gen_epochs: int = 200,
        gen_batch: int = 256,
        gen_lr: float = 1e-3,
        gen_alpha_dist: float = 1.0,
        ref_stats: Optional[Dict[str, torch.Tensor]] = None,
        X_train_ref: Optional[torch.Tensor] = None,
        modality_mask_ref: Optional[torch.Tensor] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=5, verbose=False
        )

        self.lambda_smooth = float(lambda_smooth)
        self.lambda_expert = float(lambda_expert)
        self.expert_rules = expert_rules or {}
        self.feat2idx = feat2idx or {}
        self.important_idx = _resolve_important_feature_indices(self.expert_rules, self.feat2idx)

        self.ig_steps = int(ig_steps)
        self.latent_dim = int(latent_dim)
        self.extreme_dim = int(extreme_dim)

        self.G: Optional[TabularGeneratorWithRealInput] = None
        self.ref_stats = ref_stats
        self.gen_epochs = int(gen_epochs)
        self.gen_batch = int(gen_batch)
        self.gen_lr = float(gen_lr)
        self.gen_alpha_dist = float(gen_alpha_dist)
        self.X_train_ref = X_train_ref
        self.modality_mask_ref = modality_mask_ref
        self.ig_batch_samples = int(ig_batch_samples)
        self.ig_time_subsample = int(ig_time_subsample) if ig_time_subsample else None

    def fit_generator_if_needed(self):
        """Lazily fit generator only when needed (expert penalty > 0)."""
        if self.lambda_expert <= 0:
            return  # No expert penalty, generator not required
        if self.G is not None:
            return

        # Try to load checkpoint if exists
        import os
        ckpt_path = "texgisa_generator.pt"
        if os.path.exists(ckpt_path):
            try:
                self.G = TabularGeneratorWithRealInput(
                    self.X_train_ref.shape[1],
                    self.latent_dim,
                    self.extreme_dim
                ).to(self.device)
                self.G.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
                # Also need to compute ref_stats even when loading checkpoint
                Xtr = self.X_train_ref.to(self.device)
                mu, std = _standardize_fit(Xtr)
                self.ref_stats = {"mu": mu.to(self.device), "std": std.to(self.device)}
                print("Loaded generator from checkpoint")
                return
            except Exception:
                self.G = None  # Load failed, train from scratch

        assert self.X_train_ref is not None, "X_train_ref must be provided to fit generator."
        Xtr = self.X_train_ref.to(self.device)
        mu, std = _standardize_fit(Xtr)
        self.G, self.ref_stats = train_adv_generator(
            self.model, Xtr, mu, std,
            epochs=self.gen_epochs, batch_size=self.gen_batch,
            latent_dim=self.latent_dim, extreme_dim=self.extreme_dim,
            lr=self.gen_lr, alpha_dist=self.gen_alpha_dist,
            device=self.device,
            modality_mask=self.modality_mask_ref
        )

    def step(self, X, y, m, modality_mask=None) -> Tuple[float, float, float]:
        """
        Training step with TEXGI computation.

        Returns
        -------
        loss_main : float
            Main BCE loss
        loss_smooth : float
            Temporal smoothness loss
        loss_expert : float
            Expert penalty loss
        """
        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        m = m.to(self.device, non_blocking=True)
        mask_t = modality_mask.to(self.device, non_blocking=True) if modality_mask is not None else None

        self.model.train()

        # Forward pass
        if mask_t is not None and hasattr(self.model, 'forward'):
            try:
                hazards = self.model(X, modality_mask=mask_t)
            except:
                hazards = self.model(X)
        else:
            hazards = self.model(X)

        # Main loss
        eps = 1e-6
        hazards_clamped = hazards.clamp(eps, 1 - eps)
        bce = -(y * torch.log(hazards_clamped) + (1.0 - y) * torch.log(1.0 - hazards_clamped))
        masked = bce * m
        loss_main = masked.sum() / m.sum().clamp_min(1.0)

        loss_smooth = hazards.new_tensor(0.0)
        loss_expert = hazards.new_tensor(0.0)

        # Compute TEXGI if needed
        need_phi = (self.lambda_smooth > 0) or (self.lambda_expert > 0)
        if need_phi:
            # Generator only required when expert penalty is active
            if self.lambda_expert > 0:
                self.fit_generator_if_needed()

            with torch.enable_grad():
                B = X.shape[0]
                # Subsample batch for efficiency
                sub = min(B, self.ig_batch_samples)
                idx = torch.randperm(B, device=hazards.device)[:sub]
                Xsub = X[idx].detach().clone().requires_grad_(True)

                mask_sub = mask_t[idx].detach() if mask_t is not None else None
                forward_kwargs = {"modality_mask": mask_sub} if mask_sub is not None else None

                # Compute TEXGI
                phi = texgi_time_series(
                    self.model,
                    Xsub,
                    self.G,
                    self.ref_stats if (self.G is not None and self.ref_stats is not None) else {},
                    M=self.ig_steps,
                    latent_dim=self.latent_dim,
                    extreme_dim=self.extreme_dim,
                    t_sample=self.ig_time_subsample,
                    forward_kwargs=forward_kwargs,
                )

                if self.lambda_smooth > 0:
                    omega_smooth = attribution_temporal_l1(phi)
                    loss_smooth = self.lambda_smooth * omega_smooth

                if self.lambda_expert > 0:
                    omega_expert = expert_penalty(phi, self.important_idx)
                    loss_expert = self.lambda_expert * omega_expert

        loss = loss_main + loss_smooth + loss_expert

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.opt.step()

        return loss_main.item(), loss_smooth.item(), loss_expert.item()

    @torch.no_grad()
    def evaluate_cindex(self, X_val, y_val, m_val, durations, events, modality_mask=None) -> float:
        """Fast vectorized C-index evaluation."""
        self.model.eval()

        X_val = X_val.to(self.device)
        if modality_mask is not None:
            mask_t = modality_mask.to(self.device)
            try:
                hazards = self.model(X_val, modality_mask=mask_t)
            except:
                hazards = self.model(X_val)
        else:
            hazards = self.model(X_val)

        risk = torch.sum(hazards, dim=1).cpu()  # Simple risk proxy

        # Fast vectorized C-index computation
        return cindex_fast_torch(durations, events, risk).item()


def cindex_fast_torch(durations, events, risks) -> torch.Tensor:
    """
    Fast vectorized C-index computation.

    Parameters
    ----------
    durations : array-like
        Event times
    events : array-like
        Event indicators (1 = event, 0 = censored)
    risks : array-like
        Risk scores (higher = earlier event predicted)

    Returns
    -------
    float
        Concordance index
    """
    if not torch.is_tensor(durations):
        durations = torch.tensor(durations, dtype=torch.float32)
    if not torch.is_tensor(events):
        events = torch.tensor(events, dtype=torch.float32)
    if not torch.is_tensor(risks):
        risks = torch.tensor(risks, dtype=torch.float32)

    n = len(durations)
    if n < 2:
        return torch.tensor(0.5)

    # Create pairwise comparison matrices
    dur_i = durations.unsqueeze(1)  # [n, 1]
    dur_j = durations.unsqueeze(0)  # [1, n]
    evt_i = events.unsqueeze(1)     # [n, 1]

    # Valid pairs: i observed and t_i < t_j
    valid = (evt_i == 1) & (dur_i < dur_j)

    # Risk comparisons
    risk_i = risks.unsqueeze(1)  # [n, 1]
    risk_j = risks.unsqueeze(0)  # [1, n]

    # Count concordant, discordant, and tied pairs
    concordant = valid & (risk_i > risk_j)
    discordant = valid & (risk_i < risk_j)
    tied = valid & (risk_i == risk_j)

    num_concordant = concordant.sum()
    num_discordant = discordant.sum()
    num_tied = tied.sum()

    num_pairs = num_concordant + num_discordant + num_tied

    if num_pairs == 0:
        return torch.tensor(0.5)

    cindex = (num_concordant + 0.5 * num_tied) / num_pairs
    return cindex


# ---------------------- Data Processing Utilities ----------------------

def make_intervals(df, duration_col='duration', event_col='event', n_bins=30, method='quantile'):
    """
    Discretize survival times into intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    duration_col : str
        Duration column name
    event_col : str
        Event column name
    n_bins : int
        Number of bins
    method : str
        'quantile' or 'uniform'

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'interval_number' column
    """
    import pandas as pd

    df = df.copy()
    durations = df[duration_col].values
    events = df[event_col].values

    # Only use observed events for quantiles
    observed_durations = durations[events == 1]

    if len(observed_durations) == 0:
        # Fall back to all durations if no events
        observed_durations = durations

    if method == 'quantile':
        # Use quantiles of observed events
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.unique(np.percentile(observed_durations, quantiles * 100))
    else:
        # Uniform bins
        bins = np.linspace(observed_durations.min(), observed_durations.max(), n_bins + 1)

    # Handle duplicate edges
    if len(bins) != n_bins + 1:
        # Fall back to uniform if quantiles give duplicates
        bins = np.linspace(durations.min(), durations.max(), n_bins + 1)

    # Discretize
    intervals = np.digitize(durations, bins[1:])
    intervals = np.clip(intervals, 0, n_bins - 1)

    df['interval_number'] = intervals
    return df


def build_supervision(intervals, events, num_bins):
    """
    Create one-hot supervision labels and masks.

    Parameters
    ----------
    intervals : array-like
        Discretized time intervals
    events : array-like
        Event indicators
    num_bins : int
        Number of time bins

    Returns
    -------
    labels : np.ndarray
        One-hot labels [N, T]
    masks : np.ndarray
        Risk set masks [N, T]
    """
    n_samples = len(intervals)
    labels = np.zeros((n_samples, num_bins), dtype=np.float32)
    masks = np.zeros((n_samples, num_bins), dtype=np.float32)

    for i in range(n_samples):
        t_idx = intervals[i]
        # Mask all time bins up to and including the event/censoring time
        masks[i, :t_idx+1] = 1
        # If event occurred, set hazard to 1 at that time
        if events[i] == 1:
            labels[i, t_idx] = 1

    return labels, masks