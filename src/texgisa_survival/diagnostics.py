# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Model diagnostics and evaluation utilities for survival analysis.

This module provides functions for:
- Discrete-time Brier score calculation
- Model calibration assessment
- Survival curve visualization
- Feature importance analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any


def brier_score_discrete(hazards: np.ndarray,
                        intervals: np.ndarray,
                        events: np.ndarray) -> np.ndarray:
    """
    Calculate approximate Brier score per time bin for discrete-time hazards.

    Uses target y_k = 1 if event occurred by k, else 0; predict p_k = 1 - S_k
    where S_k = prod_{j<=k} (1 - h_j).

    Parameters
    ----------
    hazards : np.ndarray of shape (n_samples, n_bins)
        Predicted hazard probabilities per time bin
    intervals : np.ndarray of shape (n_samples,)
        Discretized time intervals (1-indexed bin number)
    events : np.ndarray of shape (n_samples,)
        Event indicators (1=event, 0=censored)

    Returns
    -------
    brier_scores : np.ndarray of shape (n_bins,)
        Mean squared error at each time bin
    """
    N, T = hazards.shape
    one_minus_h = 1.0 - hazards
    S = np.cumprod(one_minus_h, axis=1)
    P = 1.0 - S  # cumulative event prob by k

    bs = np.zeros(T, dtype=np.float64)
    cnt = np.zeros(T, dtype=np.int64)

    for i in range(N):
        k = int(intervals[i])
        e = int(events[i])
        if k <= 0:
            continue
        for j in range(T):
            y = 1 if (e == 1 and (j + 1) >= k) else 0
            bs[j] += (P[i, j] - y) ** 2
            cnt[j] += 1

    bs = np.divide(bs, np.maximum(cnt, 1))
    return bs


def calibration_by_quantile(hazards: np.ndarray,
                           intervals: np.ndarray,
                           events: np.ndarray,
                           k_idx: int,
                           n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate decile calibration for cumulative event probability by a specific time bin.

    Parameters
    ----------
    hazards : np.ndarray of shape (n_samples, n_time_bins)
        Predicted hazard probabilities
    intervals : np.ndarray of shape (n_samples,)
        Discretized time intervals (1-indexed)
    events : np.ndarray of shape (n_samples,)
        Event indicators
    k_idx : int
        Time bin index (0-indexed) to evaluate calibration
    n_bins : int, default 10
        Number of calibration bins (quantiles)

    Returns
    -------
    pred_means : np.ndarray
        Mean predicted probability per calibration bin
    obs_rates : np.ndarray
        Observed event rate per calibration bin
    group_sizes : np.ndarray
        Number of samples per calibration bin
    """
    one_minus_h = 1.0 - hazards
    S = np.cumprod(one_minus_h, axis=1)
    Pk = 1.0 - S[:, k_idx]

    order = np.argsort(Pk)
    groups = np.array_split(order, n_bins)

    xs, ys, ns = [], [], []
    for g in groups:
        if len(g) == 0:
            continue
        pred_mean = Pk[g].mean()
        obs = ((events[g] == 1) & ((intervals[g] - 1) <= k_idx)).astype(np.float32).mean()
        xs.append(pred_mean)
        ys.append(obs)
        ns.append(len(g))

    return np.array(xs), np.array(ys), np.array(ns)


def hazards_to_survival(hazards: np.ndarray) -> np.ndarray:
    """
    Convert discrete hazards to survival curves.

    S_t = prod_{k<=t} (1 - h_k)

    Parameters
    ----------
    hazards : np.ndarray of shape (n_samples, n_bins)
        Predicted hazard probabilities

    Returns
    -------
    survival : np.ndarray of shape (n_samples, n_bins)
        Survival probabilities
    """
    h = np.asarray(hazards, dtype=np.float32)
    return np.cumprod(1.0 - h, axis=1)


def integrated_brier_score_discrete(hazards: np.ndarray,
                                   labels: np.ndarray,
                                   masks: np.ndarray) -> float:
    """
    Compute Integrated Brier Score over discrete time bins.

    Parameters
    ----------
    hazards : np.ndarray of shape (n_samples, n_bins)
        Predicted hazard probabilities per time bin
    labels : np.ndarray of shape (n_samples, n_bins)
        One-hot event indicators
    masks : np.ndarray of shape (n_samples, n_bins)
        Risk-set masks indicating valid time bins

    Returns
    -------
    ibs : float
        Integrated Brier Score
    """
    surv_prob = np.cumprod(1 - hazards, axis=1)

    # Ground truth survival indicator
    gt_surv = np.ones_like(hazards)
    event_bins = labels.argmax(axis=1)
    has_event = labels.sum(axis=1) > 0

    for i in np.where(has_event)[0]:
        k = event_bins[i]
        gt_surv[i, k:] = 0.0

    brier = (surv_prob - gt_surv) ** 2
    denom = masks.sum()
    if denom < 1.0:
        denom = 1.0

    return float((brier * masks).sum() / denom)


def integrated_nll_discrete(hazards: np.ndarray,
                           labels: np.ndarray,
                           masks: np.ndarray) -> float:
    """
    Compute Integrated Negative Log-Likelihood using discrete-time hazards.

    Parameters
    ----------
    hazards : np.ndarray of shape (n_samples, n_bins)
        Predicted hazard probabilities
    labels : np.ndarray of shape (n_samples, n_bins)
        One-hot event indicators
    masks : np.ndarray of shape (n_samples, n_bins)
        Risk-set masks

    Returns
    -------
    inll : float
        Integrated Negative Log-Likelihood
    """
    eps = 1e-6
    hazards = np.clip(hazards, eps, 1 - eps)
    nll = -(labels * np.log(hazards) + (1 - labels) * np.log(1 - hazards))
    denom = masks.sum()
    if denom < 1.0:
        denom = 1.0

    return float((nll * masks).sum() / denom)


def compute_diagnostics(model,
                       X: np.ndarray,
                       time: np.ndarray,
                       event: np.ndarray,
                       n_bins: int = 30) -> Dict[str, Any]:
    """
    Compute comprehensive model diagnostics.

    Parameters
    ----------
    model : object
        Trained survival model with predict_survival and predict_risk methods
    X : np.ndarray
        Feature matrix
    time : np.ndarray
        Survival times
    event : np.ndarray
        Event indicators
    n_bins : int, default 30
        Number of time bins for discretization

    Returns
    -------
    diagnostics : dict
        Dictionary containing:
        - c_index: Concordance index
        - brier_scores: Per-bin Brier scores
        - mean_brier: Mean Brier score
        - calibration: Calibration data at median bin
        - risk_scores: Predicted risk scores
        - survival_curves: Sample survival curves
    """
    from .metrics import concordance_index

    # Get predictions
    risk_scores = model.predict_risk(X)
    survival_probs = model.predict_survival(X)

    # C-index
    c_index = concordance_index(time, risk_scores, event)

    # Discretize times
    bins = np.linspace(time.min(), time.max(), n_bins + 1)
    intervals = np.digitize(time, bins)
    intervals = np.clip(intervals, 1, n_bins)

    # Convert survival to hazards for Brier calculation
    hazards = np.zeros_like(survival_probs)
    hazards[:, 0] = 1 - survival_probs[:, 0]
    for t in range(1, survival_probs.shape[1]):
        safe_denom = np.maximum(survival_probs[:, t-1], 1e-10)
        hazards[:, t] = 1 - survival_probs[:, t] / safe_denom

    # Brier scores
    brier_scores = brier_score_discrete(hazards, intervals, event)

    # Calibration at median bin
    k_idx = n_bins // 2 - 1
    pred_means, obs_rates, group_sizes = calibration_by_quantile(
        hazards, intervals, event, k_idx, n_bins=10
    )

    return {
        'c_index': c_index,
        'brier_scores': brier_scores,
        'mean_brier': float(brier_scores.mean()),
        'calibration': {
            'bin_index': k_idx,
            'pred_means': pred_means,
            'obs_rates': obs_rates,
            'group_sizes': group_sizes
        },
        'risk_scores': risk_scores,
        'survival_curves': survival_probs,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_bins': n_bins
    }


def plot_diagnostics(diagnostics: Dict[str, Any],
                    save_dir: Optional[str] = None,
                    show: bool = True) -> None:
    """
    Plot model diagnostic visualizations.

    Parameters
    ----------
    diagnostics : dict
        Output from compute_diagnostics()
    save_dir : str, optional
        Directory to save plots
    show : bool, default True
        Whether to display plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for plotting. Install with: pip install matplotlib")
        return

    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    n_bins = diagnostics['n_bins']

    # 1. Brier score over time
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, n_bins + 1), diagnostics['brier_scores'])
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Brier score')
    ax.set_title('Brier Score over Time')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'brier_over_time.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 2. Risk distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(diagnostics['risk_scores'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Risk score')
    ax.set_ylabel('Count')
    ax.set_title('Risk Score Distribution')
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'risk_distribution.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 3. Sample survival curves
    fig, ax = plt.subplots(figsize=(8, 5))
    n_samples = min(12, len(diagnostics['survival_curves']))
    indices = np.random.choice(len(diagnostics['survival_curves']), n_samples, replace=False)
    for i in indices:
        ax.plot(np.arange(1, diagnostics['survival_curves'].shape[1] + 1),
               diagnostics['survival_curves'][i], alpha=0.7)
    ax.set_xlabel('Time bin')
    ax.set_ylabel('Survival probability S(t)')
    ax.set_title('Sample Survival Curves')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'sample_survival_curves.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    # 4. Calibration plot
    calib = diagnostics['calibration']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(calib['pred_means'], calib['obs_rates'], s=50)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.set_xlabel('Predicted event probability')
    ax.set_ylabel('Observed event rate')
    ax.set_title(f'Calibration at Time Bin {calib["bin_index"] + 1}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'calibration_scatter.png'), dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"Diagnostics Summary:")
    print(f"  C-index: {diagnostics['c_index']:.4f}")
    print(f"  Mean Brier Score: {diagnostics['mean_brier']:.4f}")
    print(f"  Samples: {diagnostics['n_samples']}")
    print(f"  Features: {diagnostics['n_features']}")
