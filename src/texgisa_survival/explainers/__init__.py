# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Explainers and interpretability utilities for survival analysis models.
"""

from typing import Optional, Dict, Tuple, Union, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from .texgi import TEXGIExplainer as TEXGI_Core

# Keep existing TEXGI explainer implementation
class TEXGIExplainer_Old:
    """
    Time-dependent EXtreme Gradient Integration for survival models.
    
    This class implements the TEXGI method for computing time-dependent
    feature importance in survival analysis models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 baseline_generator: Optional[nn.Module] = None,
                 n_steps: int = 50,
                 device: str = 'cpu'):
        """
        Initialize TEXGI explainer.
        
        Parameters
        ----------
        model : nn.Module
            Trained survival model
        baseline_generator : nn.Module, optional
            Generator for adversarial baselines
        n_steps : int
            Number of integration steps
        device : str
            Device for computation
        """
        self.model = model
        self.baseline_generator = baseline_generator
        self.n_steps = n_steps
        self.device = device
        
    def explain(self,
                X: np.ndarray,
                time_indices: Optional[List[int]] = None,
                baseline: Optional[str] = 'zero',
                batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Compute TEXGI attributions.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features
        time_indices : list of int, optional
            Time bins to compute attributions for
        baseline : str or np.ndarray
            Baseline for integration ('zero', 'mean', 'adversarial', or custom array)
        batch_size : int
            Batch size for processing
            
        Returns
        -------
        attributions : dict
            Dictionary with attribution results
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        n_samples, n_features = X_tensor.shape
        
        # Determine output dimensions
        with torch.no_grad():
            sample_output = self.model(X_tensor[:1])
            n_time_bins = sample_output.shape[1] if len(sample_output.shape) > 1 else 1
        
        if time_indices is None:
            time_indices = list(range(n_time_bins))
        
        # Create baseline
        if isinstance(baseline, str):
            if baseline == 'zero':
                baseline_tensor = torch.zeros_like(X_tensor)
            elif baseline == 'mean':
                baseline_tensor = X_tensor.mean(dim=0, keepdim=True).expand_as(X_tensor)
            elif baseline == 'adversarial' and self.baseline_generator is not None:
                baseline_tensor = self._generate_adversarial_baseline(X_tensor)
            else:
                baseline_tensor = torch.zeros_like(X_tensor)
        else:
            baseline_tensor = torch.FloatTensor(baseline).to(self.device)
            if baseline_tensor.shape[0] == 1:
                baseline_tensor = baseline_tensor.expand_as(X_tensor)
        
        # Compute attributions for each time index
        all_attributions = {}
        
        for t_idx in time_indices:
            attributions_t = self._compute_ig_for_time(
                X_tensor, baseline_tensor, t_idx, batch_size
            )
            all_attributions[f'time_{t_idx}'] = attributions_t.cpu().numpy()
        
        # Compute aggregated attributions
        if len(time_indices) > 1:
            # Average across time points
            avg_attributions = np.mean([all_attributions[f'time_{t}'] for t in time_indices], axis=0)
            all_attributions['average'] = avg_attributions
            
            # Maximum absolute attribution across time
            max_attributions = np.max(np.abs([all_attributions[f'time_{t}'] for t in time_indices]), axis=0)
            all_attributions['max_abs'] = max_attributions
        
        return all_attributions
    
    def _compute_ig_for_time(self,
                            X: torch.Tensor,
                            baseline: torch.Tensor,
                            time_idx: int,
                            batch_size: int) -> torch.Tensor:
        """Compute integrated gradients for a specific time index."""
        n_samples = X.shape[0]
        attributions = torch.zeros_like(X)
        
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]
            baseline_batch = baseline[i:end_idx]
            
            # Compute path
            X_diff = X_batch - baseline_batch
            
            # Integration steps
            batch_attributions = torch.zeros_like(X_batch)
            
            for step in range(1, self.n_steps + 1):
                alpha = step / self.n_steps
                X_interp = baseline_batch + alpha * X_diff
                X_interp.requires_grad_(True)
                
                # Forward pass
                output = self.model(X_interp)
                
                # Extract target output
                if len(output.shape) > 1:
                    target = output[:, time_idx].sum()
                else:
                    target = output.sum()
                
                # Backward pass
                grads = torch.autograd.grad(target, X_interp, create_graph=False)[0]
                batch_attributions += grads / self.n_steps
            
            # Multiply by path difference
            batch_attributions *= X_diff
            attributions[i:end_idx] = batch_attributions
        
        return attributions
    
    def _generate_adversarial_baseline(self, X: torch.Tensor) -> torch.Tensor:
        """Generate adversarial baseline using the baseline generator."""
        if self.baseline_generator is None:
            return torch.zeros_like(X)
        
        batch_size = X.shape[0]
        latent_dim = 16  # Default latent dimension
        extreme_dim = 1  # Default extreme dimension
        
        with torch.no_grad():
            z = torch.randn(batch_size, latent_dim, device=X.device)
            e = torch.rand(batch_size, extreme_dim, device=X.device)
            baseline = self.baseline_generator(X, z, e)
        
        return baseline


class PermutationExplainer:
    """
    Permutation-based feature importance for survival models.
    """
    
    def __init__(self, model, metric_func=None):
        """
        Initialize permutation explainer.
        
        Parameters
        ----------
        model : object
            Trained survival model with predict_risk method
        metric_func : callable, optional
            Metric function to use for importance calculation
        """
        self.model = model
        self.metric_func = metric_func or self._default_metric
    
    def explain(self,
                X: np.ndarray,
                y_time: np.ndarray,
                y_event: np.ndarray,
                n_repeats: int = 10,
                random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Compute permutation feature importance.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y_time : np.ndarray
            Survival times
        y_event : np.ndarray
            Event indicators
        n_repeats : int
            Number of permutation repeats
        random_state : int
            Random seed
            
        Returns
        -------
        importance : dict
            Feature importance results
        """
        np.random.seed(random_state)
        
        # Baseline score
        baseline_score = self._score_model(X, y_time, y_event)
        
        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))
        
        for repeat in range(n_repeats):
            for feat_idx in range(n_features):
                # Create permuted copy
                X_perm = X.copy()
                X_perm[:, feat_idx] = np.random.permutation(X_perm[:, feat_idx])
                
                # Score with permuted feature
                perm_score = self._score_model(X_perm, y_time, y_event)
                
                # Importance = drop in performance
                importances[repeat, feat_idx] = baseline_score - perm_score
        
        return {
            'importance': importances.mean(axis=0),
            'std': importances.std(axis=0),
            'baseline_score': baseline_score
        }
    
    def _score_model(self, X, y_time, y_event):
        """Score the model using the metric function."""
        predictions = self.model.predict_risk(X)
        return self.metric_func(y_time, y_event, predictions)
    
    def _default_metric(self, y_time, y_event, risk_scores):
        """Default concordance-based metric."""
        from ..metrics import concordance_index
        return concordance_index(y_time, y_event, risk_scores)


class SHAPExplainer:
    """
    SHAP-based explainer for survival models (wrapper).
    
    This is a wrapper that provides SHAP explanations if the shap library is available.
    """
    
    def __init__(self, model, background_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer.
        
        Parameters
        ----------
        model : object
            Trained survival model
        background_data : np.ndarray, optional
            Background data for SHAP explanations
        """
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError("SHAP library not available. Install with: pip install shap")
        
        self.model = model
        self.background_data = background_data
        self.explainer = None
        
    def explain(self,
                X: np.ndarray,
                method: str = 'tree',
                **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute SHAP explanations.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
        method : str
            SHAP method ('tree', 'linear', 'kernel')
        **kwargs
            Additional arguments for SHAP explainer
            
        Returns
        -------
        explanations : dict
            SHAP values and related information
        """
        if self.explainer is None:
            self._create_explainer(method, **kwargs)
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            # Multi-output case
            shap_values = shap_values[0]  # Take first output
        
        return {
            'shap_values': shap_values,
            'expected_value': getattr(self.explainer, 'expected_value', 0),
            'feature_importance': np.abs(shap_values).mean(axis=0)
        }
    
    def _create_explainer(self, method, **kwargs):
        """Create appropriate SHAP explainer."""
        def model_predict(X):
            return self.model.predict_risk(X)
        
        if method == 'tree':
            # For tree-based models
            self.explainer = self.shap.TreeExplainer(self.model, **kwargs)
        elif method == 'linear':
            # For linear models
            self.explainer = self.shap.LinearExplainer(self.model, self.background_data, **kwargs)
        elif method == 'kernel':
            # Model-agnostic
            background = self.background_data if self.background_data is not None else X[:100]
            self.explainer = self.shap.KernelExplainer(model_predict, background, **kwargs)
        else:
            # Default to Explainer (automatic detection)
            self.explainer = self.shap.Explainer(model_predict, self.background_data, **kwargs)


def compute_time_dependent_importance(model: nn.Module,
                                   X: np.ndarray,
                                   method: str = 'texgi',
                                   **kwargs) -> pd.DataFrame:
    """
    Compute time-dependent feature importance.
    
    Parameters
    ----------
    model : nn.Module
        Trained survival model
    X : np.ndarray
        Input features
    method : str
        Method to use ('texgi', 'permutation')
    **kwargs
        Additional arguments for the explainer
        
    Returns
    -------
    importance_df : pd.DataFrame
        Time-dependent importance results
    """
    if method == 'texgi':
        explainer = TEXGIExplainer(model, **kwargs)
        attributions = explainer.explain(X)
        
        # Convert to DataFrame
        importance_data = []
        for time_key, attr_values in attributions.items():
            if time_key.startswith('time_'):
                time_idx = int(time_key.split('_')[1])
                for feat_idx in range(attr_values.shape[1]):
                    importance_data.append({
                        'time_bin': time_idx,
                        'feature': f'feature_{feat_idx}',
                        'importance': attr_values[:, feat_idx].mean(),
                        'std': attr_values[:, feat_idx].std()
                    })
        
        return pd.DataFrame(importance_data)
    
    elif method == 'permutation':
        # Permutation importance is not naturally time-dependent
        # This would need additional implementation
        raise NotImplementedError("Time-dependent permutation importance not implemented")
    
    else:
        raise ValueError(f"Unknown method: {method}")


def aggregate_importance_across_time(importance_df: pd.DataFrame,
                                   method: str = 'mean') -> pd.DataFrame:
    """
    Aggregate time-dependent importance across time bins.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Time-dependent importance DataFrame
    method : str
        Aggregation method ('mean', 'max', 'sum')
        
    Returns
    -------
    aggregated_df : pd.DataFrame
        Aggregated importance scores
    """
    if method == 'mean':
        agg_func = 'mean'
    elif method == 'max':
        agg_func = lambda x: x.abs().max()
    elif method == 'sum':
        agg_func = 'sum'
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    aggregated = importance_df.groupby('feature').agg({
        'importance': agg_func,
        'std': 'mean'
    }).reset_index()
    
    # Sort by importance
    aggregated = aggregated.sort_values('importance', 
                                      key=abs if method == 'max' else lambda x: x,
                                      ascending=False)
    
    return aggregated.reset_index(drop=True)