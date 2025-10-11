# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Evaluation metrics for survival analysis.
"""

from typing import Optional, Union, Tuple
import numpy as np
from lifelines.utils import concordance_index as lifelines_concordance_index


def concordance_index(time: np.ndarray,
                     event: np.ndarray,
                     predicted_scores: np.ndarray,
                     tied_tol: float = 1e-8) -> float:
    """
    Calculate Harrell's concordance index (C-index).
    
    The C-index measures the probability that, for a pair of randomly chosen 
    comparable samples, the sample with the higher risk score experiences an 
    event before the sample with the lower risk score.
    
    Parameters
    ----------
    time : np.ndarray
        Observed survival times
    event : np.ndarray
        Event indicators (1 if event occurred, 0 if censored)
    predicted_scores : np.ndarray
        Predicted risk scores (higher score = higher risk)
    tied_tol : float
        Tolerance for considering scores as tied
        
    Returns
    -------
    c_index : float
        Concordance index between 0 and 1 (higher is better)
    """
    return lifelines_concordance_index(time, -predicted_scores, event)


def brier_score(time: np.ndarray,
               event: np.ndarray,
               survival_probs: np.ndarray,
               eval_times: np.ndarray) -> float:
    """
    Calculate time-dependent Brier score.
    
    The Brier score measures the mean squared difference between predicted 
    survival probabilities and the observed survival status.
    
    Parameters
    ----------
    time : np.ndarray of shape (n_samples,)
        Observed survival times
    event : np.ndarray of shape (n_samples,)
        Event indicators
    survival_probs : np.ndarray of shape (n_samples, n_times)
        Predicted survival probabilities
    eval_times : np.ndarray of shape (n_times,)
        Times at which to evaluate the score
        
    Returns
    -------
    brier : float
        Mean Brier score across all time points (lower is better)
    """
    n_samples = len(time)
    n_times = len(eval_times)
    scores = np.zeros(n_times)
    
    for t_idx, t in enumerate(eval_times):
        # Calculate weights using Kaplan-Meier for censoring
        weights = _calculate_ipcw_weights(time, event, t)
        
        # Calculate Brier score at time t
        observed = (time > t).astype(float)
        predicted = survival_probs[:, t_idx]
        
        scores[t_idx] = np.mean(weights * (observed - predicted) ** 2)
    
    return np.mean(scores)


def integrated_brier_score(time: np.ndarray,
                         event: np.ndarray,
                         survival_probs: np.ndarray,
                         eval_times: Optional[np.ndarray] = None) -> float:
    """
    Calculate integrated Brier score (IBS).
    
    The IBS integrates the time-dependent Brier score over time.
    
    Parameters
    ----------
    time : np.ndarray
        Observed survival times
    event : np.ndarray
        Event indicators
    survival_probs : np.ndarray
        Predicted survival probabilities
    eval_times : np.ndarray, optional
        Times at which to evaluate. If None, uses observed event times.
        
    Returns
    -------
    ibs : float
        Integrated Brier score (lower is better)
    """
    if eval_times is None:
        eval_times = np.sort(np.unique(time[event == 1]))
        
    scores = []
    for t_idx, t in enumerate(eval_times):
        weights = _calculate_ipcw_weights(time, event, t)
        observed = (time > t).astype(float)
        
        if t_idx < survival_probs.shape[1]:
            predicted = survival_probs[:, t_idx]
        else:
            # Extrapolate if needed
            predicted = survival_probs[:, -1]
            
        score_t = np.mean(weights * (observed - predicted) ** 2)
        scores.append(score_t)
    
    # Integrate using trapezoidal rule
    dt = np.diff(eval_times)
    ibs = np.sum(0.5 * (scores[:-1] + scores[1:]) * dt) / (eval_times[-1] - eval_times[0])
    
    return ibs


def cumulative_dynamic_auc(time: np.ndarray,
                          event: np.ndarray,
                          risk_scores: np.ndarray,
                          eval_times: np.ndarray) -> float:
    """
    Calculate cumulative/dynamic AUC for survival analysis.
    
    Parameters
    ----------
    time : np.ndarray
        Observed survival times
    event : np.ndarray
        Event indicators
    risk_scores : np.ndarray
        Predicted risk scores
    eval_times : np.ndarray
        Times at which to evaluate AUC
        
    Returns
    -------
    mean_auc : float
        Mean AUC across time points
    """
    from sklearn.metrics import roc_auc_score
    
    aucs = []
    for t in eval_times:
        # Define binary outcome at time t
        # Case: experienced event before t
        # Control: survived beyond t or censored after t
        
        mask_case = (time <= t) & (event == 1)
        mask_control = time > t
        
        if mask_case.sum() > 0 and mask_control.sum() > 0:
            y_binary = np.concatenate([
                np.ones(mask_case.sum()),
                np.zeros(mask_control.sum())
            ])
            scores_binary = np.concatenate([
                risk_scores[mask_case],
                risk_scores[mask_control]
            ])
            
            try:
                auc_t = roc_auc_score(y_binary, scores_binary)
                aucs.append(auc_t)
            except:
                pass
    
    return np.mean(aucs) if aucs else 0.5


def _calculate_ipcw_weights(time: np.ndarray,
                           event: np.ndarray,
                           eval_time: float) -> np.ndarray:
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
    
    weights = np.ones(len(time))
    
    for i in range(len(time)):
        if time[i] <= eval_time and event[i] == 1:
            # Event occurred before eval_time
            surv_prob = kmf_censor.survival_function_at_times(time[i]).values[0]
            weights[i] = 1.0 / max(surv_prob, 0.01)  # Avoid division by very small numbers
        elif time[i] > eval_time:
            # Survived beyond eval_time
            surv_prob = kmf_censor.survival_function_at_times(eval_time).values[0]
            weights[i] = 1.0 / max(surv_prob, 0.01)
        else:
            # Censored before eval_time
            weights[i] = 0
            
    return weights


def log_rank_test(time1: np.ndarray,
                  event1: np.ndarray,
                  time2: np.ndarray,
                  event2: np.ndarray) -> Tuple[float, float]:
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


def calibration_slope(predicted_risk: np.ndarray,
                     observed_risk: np.ndarray,
                     n_bins: int = 10) -> float:
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