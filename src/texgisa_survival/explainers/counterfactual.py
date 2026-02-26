# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Counterfactual explanations for survival analysis models.

Two complementary strategies are provided:

1. **Hazard-scaling** (``generate_cf_from_arrays``): deterministic or genetic
   scaling of predicted hazards to achieve a target survival extension.
2. **Gradient-based** (``generate_texgi_counterfactuals``): constrained gradient
   optimisation on the input features of a trained ``MultiTaskSurvivalNet``
   (TEXGISA/MySA) model.

A unified :class:`CounterfactualExplainer` wrapper offers a single ``.explain()``
entry point for both methods.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CFResult:
    """Result of hazard-scaling counterfactual generation."""
    table: pd.DataFrame
    summary: Dict[str, Any]
    save_path: Optional[str] = None


@dataclass
class TexgiCFResult:
    """Result of gradient-based counterfactual generation."""
    table: pd.DataFrame
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers (hazard-scaling)
# ---------------------------------------------------------------------------

def _select_interval(hazards: np.ndarray, interval: Optional[int]):
    """Pick the target hazard interval per sample."""
    if interval is None:
        peak_idx = np.argmax(hazards, axis=1)
        rows = np.arange(len(hazards))
        return hazards[rows, peak_idx], peak_idx
    idx = max(0, min(int(interval) - 1, hazards.shape[1] - 1))
    return hazards[:, idx], np.full(hazards.shape[0], idx, dtype=int)


def _target_cumhaz(cumhaz: float, horizon: int, desired_extension: float) -> float:
    """Map a desired survival extension (in intervals) to a target cumulative hazard."""
    if desired_extension <= 0 or horizon <= 0:
        return cumhaz
    scale = horizon / float(horizon + desired_extension)
    return max(cumhaz * scale, 0.0)


def _standardize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score features to balance units for distance-based search."""
    numeric = df.select_dtypes(include=[np.number])
    standardized = (numeric - numeric.mean()) / numeric.std(ddof=0)
    standardized = standardized.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return standardized


def _feature_change_suggestions(
    feature_df: pd.DataFrame,
    risks: np.ndarray,
    patient_indices: Sequence[int],
    suggestions_per_patient: int,
) -> Dict[int, Sequence[Dict[str, Any]]]:
    """Propose feature deltas by referencing lower-risk neighbours.

    For each patient, locate up to ``suggestions_per_patient`` nearest neighbours
    with strictly lower risk and translate the differences into actionable deltas.
    """
    if feature_df is None or feature_df.empty:
        return {}

    feature_df = feature_df.reset_index(drop=True)
    standardized = _standardize_features(feature_df)
    risk_arr = risks.astype(float)
    suggestions: Dict[int, Sequence[Dict[str, Any]]] = {}

    for idx in patient_indices:
        if idx >= len(feature_df) or idx < 0:
            continue

        target_risk = risk_arr[idx] if idx < risk_arr.size else risk_arr.mean()
        base_vec = standardized.iloc[idx].to_numpy(dtype=float)

        lower_mask = risk_arr < target_risk
        lower_mask[idx] = False
        if not lower_mask.any():
            candidates = np.argsort(risk_arr)
        else:
            candidates = np.where(lower_mask)[0]

        neighbor_vecs = standardized.iloc[candidates].to_numpy(dtype=float)
        if neighbor_vecs.size == 0:
            continue

        dists = np.linalg.norm(neighbor_vecs - base_vec, axis=1)
        order = np.argsort(dists)
        picked = [int(candidates[o]) for o in order[: max(1, suggestions_per_patient)]]

        patient_suggestions = []
        for rank, neighbor_idx in enumerate(picked, start=1):
            neighbor_raw = feature_df.iloc[neighbor_idx]
            diff = neighbor_raw - feature_df.iloc[idx]
            non_zero = diff.replace(0.0, np.nan).dropna()
            if non_zero.empty:
                text = "Lower-risk similar case found; maintain current plan with enhanced follow-up."
            else:
                top_feats = non_zero.abs().sort_values(ascending=False).head(3)
                parts = []
                for feat, delta in top_feats.items():
                    new_val = feature_df.loc[idx, feat] + delta
                    direction = "decrease" if delta < 0 else "increase"
                    parts.append(f"Adjust {feat} to {new_val:.3f} ({direction} by {delta:+.3f})")
                text = "; ".join(parts)

            patient_suggestions.append(
                {
                    "sample_id": idx,
                    "suggestion_rank": rank,
                    "reference_id": neighbor_idx,
                    "current_risk": float(target_risk),
                    "reference_risk": float(risk_arr[neighbor_idx] if neighbor_idx < risk_arr.size else target_risk),
                    "suggested_changes": text,
                }
            )
        suggestions[idx] = patient_suggestions

    return suggestions


def _binary_scale_search(
    haz_row: np.ndarray, target_ch: float, interval: Optional[int] = None
) -> np.ndarray:
    """Deterministically shrink hazards to hit the target cumulative hazard."""
    haz_adj = haz_row.copy()
    base_ch = float(haz_adj.sum())
    if base_ch <= 0:
        return haz_adj

    if target_ch >= base_ch:
        return haz_adj

    if interval is None:
        scale = max(target_ch / base_ch, 0.0)
        return np.clip(haz_adj * scale, a_min=0.0, a_max=None)

    idx = max(0, min(int(interval), haz_adj.shape[0] - 1))
    other_ch = float(base_ch - haz_adj[idx])
    if haz_adj[idx] <= 0:
        return haz_adj

    target_for_idx = max(target_ch - other_ch, 0.0)
    if target_for_idx <= 0:
        haz_adj[idx] = 0.0
        return haz_adj

    lo, hi = 0.0, 1.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        cand = other_ch + haz_adj[idx] * mid
        if cand > target_ch:
            hi = mid
        else:
            lo = mid
    haz_adj[idx] = haz_adj[idx] * lo
    return haz_adj


def _genetic_search(
    haz_row: np.ndarray, target_ch: float, n_gen: int = 18, pop_size: int = 32
) -> np.ndarray:
    """Fallback genetic search when deterministic scaling cannot satisfy constraints."""
    rng = np.random.default_rng()
    T = haz_row.shape[0]
    pop = rng.uniform(0.0, 1.0, size=(pop_size, T))

    def fitness(scales: np.ndarray) -> np.ndarray:
        adjusted = haz_row[None, :] * scales
        cum = adjusted.sum(axis=1)
        penalty = np.maximum(cum - target_ch, 0.0)
        return np.abs(cum - target_ch) + 5.0 * penalty

    for _ in range(n_gen):
        fit = fitness(pop)
        order = np.argsort(fit)
        pop = pop[order]
        elites = pop[: max(2, pop_size // 5)]
        children = []
        while len(children) + elites.shape[0] < pop_size:
            p1, p2 = rng.choice(elites, size=2, replace=True)
            cross = rng.uniform(0.2, 0.8)
            child = cross * p1 + (1 - cross) * p2
            mutation = rng.normal(0, 0.05, size=T)
            child = np.clip(child + mutation, 0.0, 1.0)
            children.append(child)
        pop = np.vstack([elites, np.asarray(children)])

    best_idx = int(np.argmin(fitness(pop)))
    return haz_row * pop[best_idx]


# ---------------------------------------------------------------------------
# Public API — hazard-scaling counterfactuals
# ---------------------------------------------------------------------------

def generate_cf_from_arrays(
    hazards,
    risk_scores=None,
    interval: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    patient_indices: Optional[Sequence[int]] = None,
    desired_extension: float = 1.0,
    feature_df: Optional[pd.DataFrame] = None,
    suggestions_per_patient: int = 1,
    use_genetic_backup: bool = True,
) -> CFResult:
    """Create post-hoc counterfactual suggestions from hazards.

    A deterministic scaling-based search shrinks hazards to achieve a user-defined
    survival extension. When the constraint cannot be satisfied, a lightweight
    genetic search explores interval-wise scaling factors as a backup.

    Parameters
    ----------
    hazards : array-like or torch.Tensor of shape ``(n_samples, n_intervals)``
        Predicted per-interval hazards.
    risk_scores : array-like, optional
        Per-sample risk scores. Defaults to cumulative hazard sum.
    interval : int, optional
        Target a specific interval. ``None`` picks per-sample peak.
    feature_names : sequence of str, optional
        Feature names for display.
    save_path : str, optional
        If given, save the result table as CSV.
    patient_indices : sequence of int, optional
        Indices of patients to generate CFs for.
    desired_extension : float
        Target survival extension in time-interval units.
    feature_df : DataFrame, optional
        Feature matrix for neighbour-based suggestions.
    suggestions_per_patient : int
        Number of neighbour-based suggestions per patient.
    use_genetic_backup : bool
        Fall back to genetic search if scaling is insufficient.

    Returns
    -------
    CFResult
    """
    haz = torch.as_tensor(hazards, dtype=torch.float32).cpu().numpy()
    if haz.ndim == 1:
        haz = haz[:, None]

    all_indices = list(range(haz.shape[0]))
    idx_subset = list(patient_indices) if patient_indices is not None else all_indices
    idx_subset = [i for i in idx_subset if 0 <= i < haz.shape[0]]
    if not idx_subset:
        idx_subset = [0] if haz.shape[0] else []

    risks = risk_scores
    if risks is None:
        risks = haz.sum(axis=1)
    risks = np.asarray(risks, dtype=float).flatten()

    if risks.shape[0] < haz.shape[0]:
        pad = np.repeat(risks[-1] if risks.size else 0.0, haz.shape[0] - risks.shape[0])
        risks = np.concatenate([risks, pad])

    target_vals, interval_idx = _select_interval(haz, interval)
    horizon = haz.shape[1]

    feature_cf = _feature_change_suggestions(
        feature_df=feature_df,
        risks=risks,
        patient_indices=idx_subset,
        suggestions_per_patient=max(1, suggestions_per_patient),
    )

    suggestions = []
    for i in idx_subset:
        base_row = haz[i].copy()
        base_risk = float(risks[i]) if i < risks.size else float(base_row.sum())
        base_ch = float(base_row.sum())

        target_ch = _target_cumhaz(base_ch, horizon=horizon, desired_extension=desired_extension)
        scaled_row = _binary_scale_search(base_row, target_ch, interval=interval_idx[i])

        if use_genetic_backup and float(scaled_row.sum()) > target_ch * 1.01:
            scaled_row = _genetic_search(base_row, target_ch)

        achieved_ch = float(scaled_row.sum())
        achieved_extension = horizon * max(base_ch / max(achieved_ch, 1e-6) - 1.0, 0.0)
        hazard_delta = float(base_row.sum() - scaled_row.sum())

        top_features = ""
        if feature_names:
            feats = ", ".join(feature_names[:3]) if len(feature_names) >= 3 else ", ".join(feature_names)
            top_features = f" Priority focus: {feats}."

        t_int = int(interval_idx[i] + 1)
        feature_rec_list = feature_cf.get(i) or [
            {
                "suggestion_rank": 1,
                "reference_id": None,
                "current_risk": base_risk,
                "reference_risk": base_risk,
                "suggested_changes": "No lower-risk similar cases found; consider personalized intervention.",
            }
        ]

        for rec in feature_rec_list:
            suggestions.append(
                {
                    "sample_id": i,
                    "target_interval": t_int,
                    "current_cumhaz": base_ch,
                    "target_cumhaz": target_ch,
                    "achieved_cumhaz": achieved_ch,
                    "hazard_reduction": hazard_delta,
                    "estimated_extension": achieved_extension,
                    "desired_extension": desired_extension,
                    "risk_score": base_risk,
                    "suggestion_rank": rec.get("suggestion_rank"),
                    "reference_id": rec.get("reference_id"),
                    "reference_risk": rec.get("reference_risk"),
                    "suggested_changes": rec.get("suggested_changes", ""),
                    "action": (
                        f"Target survival extension: {desired_extension:.1f} time units; "
                        f"tighten hazard at interval {t_int} to reduce cumulative risk "
                        f"from {base_ch:.3f} to {achieved_ch:.3f}. "
                        "Consider enhanced follow-up frequency, symptom monitoring, "
                        "and adherence management; if insufficient, consider multi-factor intervention."
                        + top_features
                    ),
                }
            )

    cf_table = pd.DataFrame(suggestions)
    summary = {
        "interval": interval or "peak",
        "mean_risk": float(np.mean([s.get("risk_score", 0.0) for s in suggestions])) if suggestions else 0.0,
        "mean_target_hazard": float(np.mean(target_vals)) if len(target_vals) else 0.0,
        "mean_achieved_extension": float(np.mean([s.get("estimated_extension", 0.0) for s in suggestions])) if suggestions else 0.0,
        "avg_feature_references": float(np.mean([s.get("reference_risk", 0.0) for s in suggestions])) if suggestions else 0.0,
    }

    if save_path:
        cf_table.to_csv(save_path, index=False)
    return CFResult(table=cf_table, summary=summary, save_path=save_path)


def run_cf_simulation(
    model: nn.Module,
    dataloader,
    interval: Optional[int] = None,
    save_path: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
    patient_indices: Optional[Sequence[int]] = None,
    desired_extension: float = 1.0,
) -> CFResult:
    """Compute hazards from a model and dispatch CF generation."""
    hazards_list = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and not hasattr(batch, "_fields"):
            x, _, _, mod_mask = batch
        else:
            x = batch["x"] if isinstance(batch, dict) else getattr(batch, "x")
            mod_mask = batch.get("mod_mask") if isinstance(batch, dict) else getattr(batch, "mod_mask", None)
        with torch.no_grad():
            hz = model(x, mod_mask)
        hazards_list.append(hz.cpu())

    hazards = torch.cat(hazards_list, dim=0)
    return generate_cf_from_arrays(
        hazards,
        interval=interval,
        feature_names=feature_names,
        save_path=save_path,
        patient_indices=patient_indices,
        desired_extension=desired_extension,
    )


# ---------------------------------------------------------------------------
# Internal helpers (gradient-based)
# ---------------------------------------------------------------------------

def _to_feature_frame(features, feature_names: Sequence[str]) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.reset_index(drop=True)[list(feature_names)]
    arr = np.asarray(features, dtype=float)
    return pd.DataFrame(arr, columns=list(feature_names))


def _stats_frame(stats, feature_names: Sequence[str]) -> pd.DataFrame:
    if stats is None:
        return pd.DataFrame(index=feature_names)
    if isinstance(stats, pd.DataFrame):
        if "feature" in stats.columns:
            return stats.set_index("feature").reindex(feature_names)
        return stats.reindex(feature_names)
    if isinstance(stats, dict):
        df = pd.DataFrame(stats)
        if "feature" in df.columns:
            df = df.set_index("feature")
        return df.reindex(feature_names)
    return pd.DataFrame(index=feature_names)


def _optimize_single(
    model: nn.Module,
    base: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    target_ch: float,
    steps: int = 160,
    lr: float = 0.05,
    l2_weight: float = 0.01,
    frozen_mask: Optional[torch.Tensor] = None,
):
    """Gradient-optimise a single feature vector to reduce cumulative hazard."""
    base = base.clone().detach()
    with torch.no_grad():
        base.clamp_(bounds_min, bounds_max)
    x = base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    best = x.clone().detach()
    with torch.no_grad():
        haz0 = model(x).sum().item()
    best_ch = haz0

    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        hazards = model(x)
        ch = hazards.sum()
        loss_hazard = torch.relu(ch - target_ch)
        loss_l2 = l2_weight * torch.mean((x - base) ** 2)
        loss = loss_hazard + loss_l2
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(bounds_min, bounds_max)
            if frozen_mask is not None and frozen_mask.any():
                x[..., frozen_mask] = base[..., frozen_mask]

            ch_post = model(x).sum()
            if ch_post.item() < best_ch:
                best_ch = ch_post.item()
                best = x.clone().detach()
            if ch_post.item() <= target_ch * 1.02:
                break

    with torch.no_grad():
        final_haz = model(best).detach().sum().item()
    return best, haz0, final_haz


# ---------------------------------------------------------------------------
# Public API — gradient-based counterfactuals
# ---------------------------------------------------------------------------

def generate_texgi_counterfactuals(
    model_or_spec,
    features,
    *,
    hazards=None,
    feature_names: Optional[Sequence[str]] = None,
    feature_stats=None,
    patient_indices: Optional[Sequence[int]] = None,
    desired_extension: float = 1.0,
    steps: int = 160,
    lr: float = 0.05,
    top_k: int = 3,
    immutable_features: Optional[Sequence[str]] = None,
    time_bin_edges: Optional[Sequence[float]] = None,
    time_unit: Optional[str] = None,
) -> TexgiCFResult:
    """Generate per-patient counterfactuals with gradient-based optimisation.

    The function performs constrained gradient updates on selected patient
    feature vectors to lower cumulative hazard toward the target level implied
    by ``desired_extension``.

    Parameters
    ----------
    model_or_spec : nn.Module or dict
        Either a ready ``nn.Module`` that maps ``(B, D) -> (B, T)`` hazards,
        or a dict spec with keys ``input_dim``, ``num_bins``, and optionally
        ``state_path``, ``hidden``, ``depth``, ``dropout``, ``feature_names``.
    features : array-like or DataFrame
        Feature matrix ``(n_samples, n_features)``.
    hazards : array-like, optional
        Pre-computed hazards. If *None*, the model is called.
    feature_names : sequence of str, optional
        Feature names.  If ``model_or_spec`` is a dict with ``feature_names``
        they are used as fallback.
    feature_stats : DataFrame or dict, optional
        Per-feature statistics with ``min`` / ``max`` columns for bounds.
    patient_indices : sequence of int, optional
        Which patients to generate CFs for.
    desired_extension : float
        Target survival extension in interval units.
    steps : int
        Optimisation steps.
    lr : float
        Learning rate.
    top_k : int
        Number of top feature changes to report per patient.
    immutable_features : sequence of str, optional
        Features that may not be modified.
    time_bin_edges : sequence of float, optional
        Used to convert interval-unit extensions to real time units.
    time_unit : str, optional
        Label for the time unit.

    Returns
    -------
    TexgiCFResult
    """
    def _estimate_time_gain(extension_intervals: float) -> float:
        if time_bin_edges is None:
            return float("nan")
        edges = np.asarray(list(time_bin_edges), dtype=float)
        edges = edges[~np.isnan(edges)]
        if edges.size == 0:
            return float("nan")
        widths = np.diff(np.concatenate([[0.0], edges]))
        if widths.size == 0:
            return float("nan")
        typical_width = float(np.nanmedian(widths))
        return max(float(extension_intervals), 0.0) * typical_width

    # Resolve model
    if isinstance(model_or_spec, nn.Module):
        model = model_or_spec
        model.eval()
        num_bins_from_model = None
        if feature_names is None:
            raise ValueError("feature_names is required when model_or_spec is an nn.Module")
    else:
        spec = model_or_spec
        if feature_names is None:
            feature_names = list(spec.get("feature_names", []))
        model = _load_model_from_spec(spec)
        num_bins_from_model = int(spec.get("num_bins", 20))

    feat_names = list(feature_names)
    feat_df = _to_feature_frame(features, feat_names)
    idx_subset = list(patient_indices) if patient_indices is not None else list(range(len(feat_df)))
    idx_subset = [i for i in idx_subset if 0 <= i < len(feat_df)]
    if not idx_subset:
        return TexgiCFResult(pd.DataFrame(), {"error": "No valid patients provided."})

    stats_df = _stats_frame(feature_stats, feat_names)
    bounds_min_raw = stats_df.get("min") if "min" in stats_df else pd.Series(np.nan, index=feat_names)
    bounds_max_raw = stats_df.get("max") if "max" in stats_df else pd.Series(np.nan, index=feat_names)

    bounds_min_arr = bounds_min_raw.to_numpy(dtype=float)
    bounds_max_arr = bounds_max_raw.to_numpy(dtype=float)
    bounds_min_arr = np.where(np.isnan(bounds_min_arr), -np.inf, bounds_min_arr)
    bounds_max_arr = np.where(np.isnan(bounds_max_arr), np.inf, bounds_max_arr)
    bounds_low = np.minimum(bounds_min_arr, bounds_max_arr)
    bounds_high = np.maximum(bounds_min_arr, bounds_max_arr)

    bounds_min_t = torch.tensor(bounds_low, dtype=torch.float32)
    bounds_max_t = torch.tensor(bounds_high, dtype=torch.float32)

    immutable_set = set(immutable_features or [])
    frozen_mask = torch.tensor([name in immutable_set for name in feat_names], dtype=torch.bool)

    # Determine num_bins for horizon
    if num_bins_from_model is not None:
        num_bins = num_bins_from_model
    else:
        # Infer from a forward pass
        with torch.no_grad():
            sample_out = model(torch.zeros(1, len(feat_names), dtype=torch.float32))
            num_bins = sample_out.shape[-1] if sample_out.ndim > 1 else 1

    hazards_arr = None
    if hazards is not None:
        haz_np = torch.as_tensor(hazards, dtype=torch.float32).cpu().numpy()
        hazards_arr = haz_np if haz_np.ndim == 2 else haz_np[:, None]

    suggestions = []
    patient_notes = []
    for idx in idx_subset:
        base_vec = torch.tensor(feat_df.iloc[idx].to_numpy(dtype=float), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_haz = model(base_vec).sum().item() if hazards_arr is None else float(hazards_arr[idx].sum())

        target_ch = _target_cumhaz(base_haz, horizon=num_bins, desired_extension=desired_extension)
        best_vec, base_ch, achieved_ch = _optimize_single(
            model, base_vec, bounds_min_t, bounds_max_t, target_ch,
            steps=steps, lr=lr, frozen_mask=frozen_mask,
        )

        delta = (best_vec - base_vec).squeeze(0).detach().cpu().numpy()
        base_arr = base_vec.squeeze(0).detach().cpu().numpy()
        best_arr = best_vec.squeeze(0).detach().cpu().numpy()
        order = np.argsort(np.abs(delta))[::-1]

        any_suggestion = False
        for rank, feat_idx in enumerate(order[: max(1, top_k)], start=1):
            if frozen_mask[feat_idx]:
                continue
            if np.isclose(delta[feat_idx], 0.0):
                continue
            feat_name = feat_names[feat_idx]
            est_gain_intervals = float(num_bins * max(base_ch / max(achieved_ch, 1e-6) - 1.0, 0.0))
            suggestions.append(
                {
                    "sample_id": idx,
                    "suggestion_rank": rank,
                    "feature": feat_name,
                    "current_value": float(base_arr[feat_idx]),
                    "suggested_value": float(best_arr[feat_idx]),
                    "delta": float(delta[feat_idx]),
                    "current_cumhaz": float(base_ch),
                    "target_cumhaz": float(target_ch),
                    "achieved_cumhaz": float(achieved_ch),
                    "estimated_extension": est_gain_intervals,
                    "estimated_extension_time": _estimate_time_gain(est_gain_intervals),
                    "time_unit": time_unit,
                }
            )
            any_suggestion = True

        if not any_suggestion:
            reason_parts = []
            if bool(frozen_mask.all()):
                reason_parts.append("All features have been locked")
            if base_ch <= target_ch:
                reason_parts.append("The current cumulative risk has fallen below the target")
            if not reason_parts:
                reason_parts.append("Gradient optimization did not find feasible updates")
            patient_notes.append(
                {
                    "sample_id": idx,
                    "base_cumhaz": float(base_ch),
                    "target_cumhaz": float(target_ch),
                    "achieved_cumhaz": float(achieved_ch),
                    "note": "; ".join(reason_parts),
                }
            )

    table = pd.DataFrame(suggestions)
    summary = {
        "desired_extension": desired_extension,
        "mean_current_cumhaz": float(table["current_cumhaz"].mean()) if not table.empty else 0.0,
        "mean_target_cumhaz": float(table["target_cumhaz"].mean()) if not table.empty else 0.0,
        "mean_achieved_cumhaz": float(table["achieved_cumhaz"].mean()) if not table.empty else 0.0,
        "mean_estimated_extension": float(table["estimated_extension"].mean()) if not table.empty else 0.0,
        "mean_estimated_time_gain": float(table["estimated_extension_time"].mean()) if not table.empty else float("nan"),
        "time_unit": time_unit,
        "patient_notes": patient_notes,
    }
    return TexgiCFResult(table=table, summary=summary)


def _load_model_from_spec(spec: Dict[str, Any]) -> nn.Module:
    """Instantiate a MultiTaskSurvivalNet from a dict specification."""
    from ..models.texgisa import MultiTaskSurvivalNet

    model = MultiTaskSurvivalNet(
        input_dim=int(spec["input_dim"]),
        num_time_bins=int(spec.get("num_bins", 20)),
        hidden_layers=[int(spec.get("hidden", 256))] * int(spec.get("depth", 2)),
        dropout=float(spec.get("dropout", 0.2)),
    )
    state_path = spec.get("state_path")
    if state_path:
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

class CounterfactualExplainer:
    """Unified interface for counterfactual explanations.

    Parameters
    ----------
    model : nn.Module, optional
        A model that maps ``(B, D) -> (B, T)`` hazards.  Required for
        ``method='gradient'``.
    feature_names : sequence of str, optional
        Feature names. Required for ``method='gradient'``.

    Examples
    --------
    >>> explainer = CounterfactualExplainer(model=my_model, feature_names=names)
    >>> result = explainer.explain(features, method='gradient', desired_extension=2.0)
    >>> result = explainer.explain(hazards=haz_array, method='scaling', desired_extension=1.5)
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None

    def explain(
        self,
        features=None,
        *,
        hazards=None,
        method: str = "scaling",
        desired_extension: float = 1.0,
        patient_indices: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        """Generate counterfactual explanations.

        Parameters
        ----------
        features : array-like, optional
            Feature matrix. Required for ``method='gradient'``.
        hazards : array-like, optional
            Pre-computed hazards. Used by ``method='scaling'`` and optionally
            by ``method='gradient'``.
        method : ``'scaling'`` or ``'gradient'``
            Which strategy to use.
        desired_extension : float
            Target survival extension (in interval units).
        patient_indices : sequence of int, optional
            Subset of patients to explain.
        **kwargs
            Forwarded to the underlying function.

        Returns
        -------
        CFResult or TexgiCFResult
        """
        if method == "scaling":
            if hazards is None:
                if self.model is None or features is None:
                    raise ValueError("Either hazards or (model + features) must be provided for scaling method.")
                features_t = torch.as_tensor(np.asarray(features, dtype="float32"))
                with torch.no_grad():
                    hazards = self.model(features_t).cpu()
            return generate_cf_from_arrays(
                hazards=hazards,
                patient_indices=patient_indices,
                desired_extension=desired_extension,
                feature_names=self.feature_names,
                feature_df=pd.DataFrame(np.asarray(features), columns=self.feature_names) if features is not None and self.feature_names else None,
                **kwargs,
            )
        elif method == "gradient":
            if self.model is None:
                raise ValueError("A model is required for gradient-based counterfactuals.")
            if features is None:
                raise ValueError("Features are required for gradient-based counterfactuals.")
            if self.feature_names is None:
                raise ValueError("feature_names is required for gradient-based counterfactuals.")
            return generate_texgi_counterfactuals(
                model_or_spec=self.model,
                features=features,
                hazards=hazards,
                feature_names=self.feature_names,
                patient_indices=patient_indices,
                desired_extension=desired_extension,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'scaling' or 'gradient'.")
