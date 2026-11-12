# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Evaluation metrics for survival analysis.
"""

from typing import Optional, Union, Tuple
import numpy as np
from lifelines.utils import concordance_index as lifelines_concordance_index


def concordance_index(
    time: np.ndarray, predicted_scores: np.ndarray, event: np.ndarray, tied_tol: float = 1e-8
) -> float:
    """
    Calculate Harrell's concordance index (C-index).

    The C-index measures the probability that, for a pair of randomly chosen
    comparable samples, the sample with the higher risk score experiences an
    event before the sample with the lower risk score.

    Parameters
    ----------
    time : np.ndarray
        Observed survival times
    predicted_scores : np.ndarray
        Predicted risk scores (higher score = higher risk)
    event : np.ndarray
        Event indicators (1 if event occurred, 0 if censored)
    tied_tol : float
        Tolerance for considering scores as tied

    Returns
    -------
    c_index : float
        Concordance index between 0 and 1 (higher is better)
    """
    # Convert to numpy arrays if needed
    time = np.asarray(time)
    predicted_scores = np.asarray(predicted_scores)
    event = np.asarray(event)

    # Validate input lengths
    if not (len(time) == len(predicted_scores) == len(event)):
        raise ValueError("time, predicted_scores, and event must have the same length")

    # Our API: predicted_scores are risk scores (higher = earlier event), as
    # returned by predict_risk(). lifelines expects the opposite orientation
    # (higher = longer survival), so the scores are negated before the call.
    return lifelines_concordance_index(time, -predicted_scores, event)


def brier_score(
    time: np.ndarray,
    survival_probs: np.ndarray,
    event: np.ndarray,
    time_point: float = None,
    eval_times: np.ndarray = None,
) -> float:
    """
    Calculate time-dependent Brier score.

    The Brier score measures the mean squared difference between predicted
    survival probabilities and the observed survival status.

    Parameters
    ----------
    time : np.ndarray of shape (n_samples,)
        Observed survival times
    survival_probs : np.ndarray of shape (n_samples,) or (n_samples, n_times)
        Predicted survival probabilities
    event : np.ndarray of shape (n_samples,)
        Event indicators
    time_point : float, optional
        Single time point at which to evaluate (for backward compatibility)
    eval_times : np.ndarray of shape (n_times,), optional
        Times at which to evaluate the score

    Returns
    -------
    brier : float
        Mean Brier score across all time points (lower is better)
    """
    time = np.asarray(time)
    survival_probs = np.asarray(survival_probs)
    event = np.asarray(event)

    # Handle single time point evaluation
    if time_point is not None:
        if survival_probs.ndim == 2 and survival_probs.shape[1] != 1:
            raise ValueError(
                "With time_point, survival_probs must be 1-D or have one column "
                "(the predicted survival at time_point)."
            )
        predicted = survival_probs.reshape(-1)
        weights = _calculate_ipcw_weights(time, event, time_point)
        observed = (time > time_point).astype(float)
        return float(np.mean(weights * (observed - predicted) ** 2))

    # Multiple time points
    if eval_times is None:
        eval_times = np.sort(np.unique(time[event == 1]))
    eval_times = np.asarray(eval_times)
    _check_survival_grid(survival_probs, eval_times)

    scores = np.zeros(len(eval_times))
    for t_idx, t in enumerate(eval_times):
        weights = _calculate_ipcw_weights(time, event, t)
        observed = (time > t).astype(float)
        predicted = survival_probs.reshape(len(time), -1)[:, t_idx]
        scores[t_idx] = np.mean(weights * (observed - predicted) ** 2)

    return float(np.mean(scores))


def _check_survival_grid(survival_probs: np.ndarray, eval_times: np.ndarray) -> None:
    """Column k of survival_probs must be the prediction at eval_times[k]."""
    n_cols = 1 if survival_probs.ndim == 1 else survival_probs.shape[1]
    if n_cols != len(eval_times):
        raise ValueError(
            f"survival_probs has {n_cols} column(s) but {len(eval_times)} evaluation times were "
            "given; column k must hold S(eval_times[k]). Use "
            "model.predict_survival(X, times=eval_times) to obtain matching predictions."
        )


def integrated_brier_score(
    time: np.ndarray,
    survival_probs: np.ndarray,
    event: np.ndarray,
    eval_times: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate integrated Brier score (IBS).

    The IBS integrates the time-dependent Brier score over time.

    Parameters
    ----------
    time : np.ndarray
        Observed survival times
    survival_probs : np.ndarray
        Predicted survival probabilities
    event : np.ndarray
        Event indicators
    eval_times : np.ndarray, optional
        Times at which to evaluate. If None, uses observed event times.

    Returns
    -------
    ibs : float
        Integrated Brier score (lower is better)
    """
    time = np.asarray(time)
    survival_probs = np.asarray(survival_probs)
    event = np.asarray(event)

    if eval_times is None:
        eval_times = np.sort(np.unique(time[event == 1]))
    else:
        eval_times = np.asarray(eval_times)

    _check_survival_grid(survival_probs, eval_times)
    if len(eval_times) == 1:
        # Single time point - just return the Brier score
        return brier_score(time, survival_probs, event, time_point=eval_times[0])

    S = survival_probs.reshape(len(time), -1)
    scores = []
    for t_idx, t in enumerate(eval_times):
        weights = _calculate_ipcw_weights(time, event, t)
        observed = (time > t).astype(float)
        score_t = np.mean(weights * (observed - S[:, t_idx]) ** 2)
        scores.append(score_t)

    # Integrate using trapezoidal rule
    scores = np.array(scores)
    dt = np.diff(eval_times)
    ibs = np.sum(0.5 * (scores[:-1] + scores[1:]) * dt) / (eval_times[-1] - eval_times[0])

    return float(ibs)


def cumulative_dynamic_auc(
    time: np.ndarray, risk_scores: np.ndarray, event: np.ndarray, eval_times: np.ndarray
) -> np.ndarray:
    """
    Calculate cumulative/dynamic AUC for survival analysis.

    Parameters
    ----------
    time : np.ndarray
        Observed survival times
    risk_scores : np.ndarray
        Predicted risk scores
    event : np.ndarray
        Event indicators
    eval_times : np.ndarray
        Times at which to evaluate AUC

    Returns
    -------
    aucs : np.ndarray
        AUC values at each time point
    """
    from sklearn.metrics import roc_auc_score

    time = np.asarray(time)
    risk_scores = np.asarray(risk_scores)
    event = np.asarray(event)
    eval_times = np.asarray(eval_times)

    aucs = []
    for t in eval_times:
        # Define binary outcome at time t
        # Case: experienced event before t
        # Control: survived beyond t or censored after t

        mask_case = (time <= t) & (event == 1)
        mask_control = time > t

        if mask_case.sum() > 0 and mask_control.sum() > 0:
            y_binary = np.concatenate([np.ones(mask_case.sum()), np.zeros(mask_control.sum())])
            scores_binary = np.concatenate([risk_scores[mask_case], risk_scores[mask_control]])

            try:
                auc_t = roc_auc_score(y_binary, scores_binary)
                aucs.append(auc_t)
            except Exception:
                aucs.append(0.5)
        else:
            aucs.append(0.5)

    return np.array(aucs)


def _calculate_ipcw_weights(time: np.ndarray, event: np.ndarray, eval_time: float) -> np.ndarray:
    """
    Calculate inverse probability of censoring weights (IPCW).

    Parameters
    ----------
    time : np.ndarray
        Observed times
    event : np.ndarray
        Event indicators
    eval_time : float
        Evaluation time point

    Returns
    -------
    weights : np.ndarray
        IPCW weights for each sample
    """
    from lifelines import KaplanMeierFitter

    # Fit KM for censoring distribution (reverse the event indicator)
    kmf_censor = KaplanMeierFitter()
    kmf_censor.fit(time, 1 - event)

    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    weights = np.zeros(len(time))  # censored before eval_time: weight 0

    # Event by eval_time: 1 / G(T_i-), the censoring survival just before T_i (Graf et al., 1999)
    had_event = (time <= eval_time) & (event == 1)
    if had_event.any():
        g = kmf_censor.survival_function_at_times(np.nextafter(time[had_event], -np.inf)).values
        weights[had_event] = 1.0 / np.maximum(g, 0.01)  # avoid division by very small numbers

    # Still at risk after eval_time: 1 / G(eval_time)
    alive = time > eval_time
    if alive.any():
        g = kmf_censor.survival_function_at_times(eval_time).values[0]
        weights[alive] = 1.0 / max(g, 0.01)

    return weights


def log_rank_test(
    time1: np.ndarray, event1: np.ndarray, time2: np.ndarray, event2: np.ndarray
) -> Tuple[float, float]:
    """
    Perform log-rank test to compare two survival curves.

    Parameters
    ----------
    time1, event1 : np.ndarray
        Survival data for group 1
    time2, event2 : np.ndarray
        Survival data for group 2

    Returns
    -------
    statistic : float
        Test statistic
    p_value : float
        P-value for the test
    """
    from lifelines.statistics import logrank_test

    result = logrank_test(time1, time2, event1, event2)
    return result.test_statistic, result.p_value


def kaplan_meier_estimator(
    durations: np.ndarray,
    events: np.ndarray,
    limit_to_last_event: bool = True,
    bin_width: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Kaplan-Meier survival curve (pure numpy, no lifelines).

    Parameters
    ----------
    durations : np.ndarray
        Observed survival / follow-up times.
    events : np.ndarray
        Event indicators (1 = event, 0 = censored).
    limit_to_last_event : bool
        If *True*, truncate the timeline at the last observed event time.
    bin_width : float
        When > 0, bucket times into bins of this width before estimation.

    Returns
    -------
    times : np.ndarray
        Time coordinates of the step-function nodes (starts at 0).
    survival_probs : np.ndarray
        Corresponding survival probabilities.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=int)

    if bin_width and bin_width > 0:
        d = (np.floor(d / bin_width) * bin_width).astype(float)

    if limit_to_last_event and (e == 1).any():
        t_last = d[e == 1].max()
        keep = d <= t_last
        d, e = d[keep], e[keep]

    order = np.argsort(d)
    d, e = d[order], e[order]

    uniq = np.unique(d)
    n_at_risk = len(d)
    S = 1.0
    t_nodes = [0.0]
    S_nodes = [1.0]

    for t in uniq:
        at_t = d == t
        m = at_t.sum()
        d_events = (e[at_t] == 1).sum()
        if d_events > 0 and n_at_risk > 0:
            S = S * (1.0 - d_events / n_at_risk)
            t_nodes.append(float(t))
            S_nodes.append(float(S))
        n_at_risk -= m

    return np.array(t_nodes), np.array(S_nodes)


def calibration_slope(
    predicted_risk: np.ndarray, observed_risk: np.ndarray, n_bins: int = 10
) -> float:
    """
    Calculate calibration slope.

    A well-calibrated model has a slope close to 1.

    Parameters
    ----------
    predicted_risk : np.ndarray
        Predicted risk scores
    observed_risk : np.ndarray
        Observed outcomes
    n_bins : int
        Number of bins for grouping predictions

    Returns
    -------
    slope : float
        Calibration slope
    """
    from sklearn.linear_model import LinearRegression

    # Bin predictions
    bins = np.percentile(predicted_risk, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    observed_freq = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (predicted_risk >= bins[i]) & (predicted_risk < bins[i + 1])
        else:
            mask = predicted_risk >= bins[i]

        if mask.sum() > 0:
            bin_centers.append(predicted_risk[mask].mean())
            observed_freq.append(observed_risk[mask].mean())

    if len(bin_centers) < 2:
        return np.nan

    # Fit linear regression
    reg = LinearRegression()
    X = np.array(bin_centers).reshape(-1, 1)
    y = np.array(observed_freq)
    reg.fit(X, y)

    return reg.coef_[0]
