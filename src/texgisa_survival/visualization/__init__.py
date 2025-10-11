# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Visualization utilities for survival analysis.
"""

from typing import Optional, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter


def plot_survival_curves(survival_probs: np.ndarray,
                        times: np.ndarray,
                        sample_indices: Optional[List[int]] = None,
                        labels: Optional[List[str]] = None,
                        title: str = "Predicted Survival Curves",
                        xlabel: str = "Time",
                        ylabel: str = "Survival Probability",
                        figsize: tuple = (10, 6),
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """
    Plot survival curves for selected samples.
    
    Parameters
    ----------
    survival_probs : np.ndarray of shape (n_samples, n_times)
        Predicted survival probabilities
    times : np.ndarray of shape (n_times,)
        Time points
    sample_indices : list of int, optional
        Indices of samples to plot. If None, plots first 5 samples.
    labels : list of str, optional
        Labels for each curve
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if sample_indices is None:
        sample_indices = list(range(min(5, survival_probs.shape[0])))
    
    if labels is None:
        labels = [f"Sample {i}" for i in sample_indices]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(sample_indices)))
    
    for i, (idx, label, color) in enumerate(zip(sample_indices, labels, colors)):
        if idx < survival_probs.shape[0]:
            ax.plot(times, survival_probs[idx, :], 
                   label=label, color=color, linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_kaplan_meier(time: np.ndarray,
                     event: np.ndarray,
                     groups: Optional[np.ndarray] = None,
                     group_labels: Optional[List[str]] = None,
                     title: str = "Kaplan-Meier Survival Curves",
                     xlabel: str = "Time",
                     ylabel: str = "Survival Probability",
                     figsize: tuple = (10, 6),
                     confidence_intervals: bool = True,
                     save_path: Optional[str] = None,
                     show: bool = True) -> plt.Figure:
    """
    Plot Kaplan-Meier survival curves.
    
    Parameters
    ----------
    time : np.ndarray
        Survival times
    event : np.ndarray
        Event indicators
    groups : np.ndarray, optional
        Group assignments for each sample
    group_labels : list of str, optional
        Labels for each group
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    confidence_intervals : bool
        Whether to show confidence intervals
    save_path : str, optional
        Path to save the plot
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if groups is None:
        # Single curve for all data
        kmf = KaplanMeierFitter()
        kmf.fit(time, event)
        kmf.plot_survival_function(ax=ax, ci_show=confidence_intervals)
    else:
        # Multiple curves for different groups
        unique_groups = np.unique(groups)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        
        if group_labels is None:
            group_labels = [f"Group {g}" for g in unique_groups]
        
        for group, label, color in zip(unique_groups, group_labels, colors):
            mask = groups == group
            if mask.sum() > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(time[mask], event[mask], label=label)
                kmf.plot_survival_function(ax=ax, ci_show=confidence_intervals, color=color)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                          top_n: int = 20,
                          title: str = "Feature Importance",
                          figsize: tuple = (10, 8),
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """
    Plot feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with columns 'feature' and 'importance'
    top_n : int
        Number of top features to display
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get top N features
    plot_data = importance_df.head(top_n).copy()
    plot_data = plot_data.sort_values('importance', ascending=True)  # For horizontal bars
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))
    
    bars = ax.barh(range(len(plot_data)), plot_data['importance'], color=colors)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data['feature'])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, plot_data['importance'])):
        ax.text(value + max(plot_data['importance']) * 0.01, i, 
                f'{value:.3f}', va='center', fontsize=9)
    
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_risk_scores(risk_scores: np.ndarray,
                    actual_times: Optional[np.ndarray] = None,
                    actual_events: Optional[np.ndarray] = None,
                    title: str = "Risk Score Distribution",
                    figsize: tuple = (12, 5),
                    save_path: Optional[str] = None,
                    show: bool = True) -> plt.Figure:
    """
    Plot distribution of risk scores.
    
    Parameters
    ----------
    risk_scores : np.ndarray
        Predicted risk scores
    actual_times : np.ndarray, optional
        Actual survival times
    actual_events : np.ndarray, optional
        Actual event indicators
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    if actual_times is not None and actual_events is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0]//2, figsize[1]))
    
    # Risk score histogram
    ax1.hist(risk_scores, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Risk Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Risk Score Distribution')
    ax1.grid(True, alpha=0.3)
    
    if actual_times is not None and actual_events is not None:
        # Risk vs survival time scatter plot
        colors = ['red' if e == 1 else 'blue' for e in actual_events]
        labels = ['Event' if e == 1 else 'Censored' for e in actual_events]
        
        for event_type in [0, 1]:
            mask = actual_events == event_type
            if mask.sum() > 0:
                label = 'Event' if event_type == 1 else 'Censored'
                color = 'red' if event_type == 1 else 'blue'
                ax2.scatter(risk_scores[mask], actual_times[mask], 
                          alpha=0.6, label=label, c=color, s=20)
        
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Actual Survival Time')
        ax2.set_title('Risk Score vs Actual Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_calibration(predicted_probs: np.ndarray,
                    actual_times: np.ndarray,
                    actual_events: np.ndarray,
                    eval_times: np.ndarray,
                    n_bins: int = 10,
                    title: str = "Calibration Plot",
                    figsize: tuple = (10, 8),
                    save_path: Optional[str] = None,
                    show: bool = True) -> plt.Figure:
    """
    Plot calibration curves for survival probabilities.
    
    Parameters
    ----------
    predicted_probs : np.ndarray of shape (n_samples, n_times)
        Predicted survival probabilities
    actual_times : np.ndarray
        Actual survival times
    actual_events : np.ndarray
        Actual event indicators
    eval_times : np.ndarray
        Times at which predictions were made
    n_bins : int
        Number of bins for calibration
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot calibration for selected time points
    time_indices = [0, len(eval_times)//3, 2*len(eval_times)//3, -1]
    
    for i, time_idx in enumerate(time_indices):
        ax = axes[i]
        time_point = eval_times[time_idx]
        
        # Get predictions for this time point
        pred_probs = predicted_probs[:, time_idx]
        
        # Calculate observed survival at this time point
        observed = (actual_times > time_point).astype(float)
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        observed_freq = []
        predicted_freq = []
        
        for j in range(n_bins):
            if j < n_bins - 1:
                mask = (pred_probs >= bin_edges[j]) & (pred_probs < bin_edges[j + 1])
            else:
                mask = pred_probs >= bin_edges[j]
            
            if mask.sum() > 0:
                observed_freq.append(observed[mask].mean())
                predicted_freq.append(pred_probs[mask].mean())
            else:
                observed_freq.append(0)
                predicted_freq.append(bin_centers[j])
        
        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.scatter(predicted_freq, observed_freq, alpha=0.7, s=50, label='Observed')
        ax.plot(predicted_freq, observed_freq, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Frequency')
        ax.set_title(f'Time = {time_point:.1f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame,
                         metric: str = 'c-index_mean',
                         title: str = "Model Comparison",
                         figsize: tuple = (10, 6),
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model comparison results
    metric : str
        Metric to plot (column name)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save the plot
    show : bool
        Whether to display the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by metric value
    plot_data = comparison_df.sort_values(metric, ascending=True)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_data)))
    
    bars = ax.barh(range(len(plot_data)), plot_data[metric], color=colors)
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data['model'])
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, plot_data[metric])):
        ax.text(value + max(plot_data[metric]) * 0.01, i,
                f'{value:.3f}', va='center', fontsize=10)
    
    # Add error bars if std column exists
    std_col = metric.replace('_mean', '_std')
    if std_col in plot_data.columns:
        ax.errorbar(plot_data[metric], range(len(plot_data)), 
                   xerr=plot_data[std_col], fmt='none', color='black', alpha=0.5)
    
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig