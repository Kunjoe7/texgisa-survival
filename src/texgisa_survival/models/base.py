# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Base class for all survival models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn


class BaseSurvivalModel(ABC):
    """Abstract base class for survival models."""
    
    def __init__(self, device: str = 'cpu', random_state: Optional[int] = None):
        """
        Initialize base survival model.
        
        Parameters
        ----------
        device : str
            Device for computation ('cpu', 'cuda', 'mps')
        random_state : int, optional
            Random seed for reproducibility
        """
        self.device = device
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
        
        if random_state is not None:
            np.random.seed(random_state)
            torch.manual_seed(random_state)
            if device == 'cuda':
                torch.cuda.manual_seed(random_state)
    
    @abstractmethod
    def fit(self,
            X: np.ndarray,
            time: np.ndarray,
            event: np.ndarray,
            validation_data: Optional[Tuple] = None,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001,
            early_stopping: bool = True,
            patience: int = 10,
            verbose: int = 1,
            **kwargs) -> None:
        """
        Train the survival model.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        time : np.ndarray of shape (n_samples,)
            Time to event or censoring
        event : np.ndarray of shape (n_samples,)
            Event indicator (1=event, 0=censored)
        validation_data : tuple, optional
            (X_val, time_val, event_val) for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate
        early_stopping : bool
            Whether to use early stopping
        patience : int
            Patience for early stopping
        verbose : int
            Verbosity level
        **kwargs : dict
            Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def predict_survival(self,
                        X: np.ndarray,
                        times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict survival probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        times : np.ndarray of shape (n_times,), optional
            Times at which to evaluate survival
            
        Returns
        -------
        survival_probs : np.ndarray of shape (n_samples, n_times)
            Survival probabilities
        """
        pass
    
    @abstractmethod
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
            
        Returns
        -------
        risk_scores : np.ndarray of shape (n_samples,)
            Risk scores
        """
        pass
    
    def predict_hazard(self,
                      X: np.ndarray,
                      times: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict hazard function.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix
        times : np.ndarray of shape (n_times,), optional
            Times at which to evaluate hazard
            
        Returns
        -------
        hazard : np.ndarray of shape (n_samples, n_times)
            Hazard values
        """
        # Default implementation using survival probabilities
        surv_probs = self.predict_survival(X, times)
        
        # Approximate hazard from survival: h(t) ≈ -d/dt log(S(t))
        hazard = np.zeros_like(surv_probs)
        eps = 1e-7
        
        # Use finite differences
        dt = np.diff(times) if times is not None else 1.0
        for i in range(surv_probs.shape[1] - 1):
            hazard[:, i] = -(np.log(surv_probs[:, i+1] + eps) - 
                           np.log(surv_probs[:, i] + eps)) / dt[i] if isinstance(dt, np.ndarray) else dt
        
        # Last time point: use previous hazard
        hazard[:, -1] = hazard[:, -2] if surv_probs.shape[1] > 1 else 0
        
        return hazard
    
    def get_feature_importance(self,
                             method: str = 'default',
                             **kwargs) -> Dict[str, np.ndarray]:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        method : str
            Method for importance calculation
        **kwargs : dict
            Additional method-specific parameters
            
        Returns
        -------
        importance : dict
            Dictionary with 'importance' and optionally 'std' arrays
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing feature importance")
        
        if method == 'default' or method == 'weights':
            return self._get_weight_importance()
        elif method == 'permutation':
            return self._get_permutation_importance(**kwargs)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _get_weight_importance(self) -> Dict[str, np.ndarray]:
        """Get importance from model weights (if applicable)."""
        if not hasattr(self.model, 'fc1'):
            raise NotImplementedError("Weight importance not available for this model")
        
        # Get first layer weights
        weights = self.model.fc1.weight.data.cpu().numpy()
        importance = np.abs(weights).mean(axis=0)
        
        return {
            'importance': importance,
            'std': np.abs(weights).std(axis=0)
        }
    
    def _get_permutation_importance(self,
                                   X: Optional[np.ndarray] = None,
                                   time: Optional[np.ndarray] = None,
                                   event: Optional[np.ndarray] = None,
                                   n_repeats: int = 10) -> Dict[str, np.ndarray]:
        """Calculate permutation importance."""
        if X is None or time is None or event is None:
            raise ValueError("X, time, and event required for permutation importance")
        
        from ..metrics import concordance_index
        
        # Baseline score
        risk_scores = self.predict_risk(X)
        baseline_score = concordance_index(time, event, risk_scores)
        
        n_features = X.shape[1]
        importances = np.zeros((n_repeats, n_features))
        
        for feat_idx in range(n_features):
            for repeat in range(n_repeats):
                # Permute feature
                X_perm = X.copy()
                X_perm[:, feat_idx] = np.random.permutation(X_perm[:, feat_idx])
                
                # Calculate score with permuted feature
                risk_scores_perm = self.predict_risk(X_perm)
                perm_score = concordance_index(time, event, risk_scores_perm)
                
                # Importance is drop in performance
                importances[repeat, feat_idx] = baseline_score - perm_score
        
        return {
            'importance': importances.mean(axis=0),
            'std': importances.std(axis=0)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get model state for saving.
        
        Returns
        -------
        state : dict
            Model state dictionary
        """
        state = {
            'is_fitted': self.is_fitted,
            'device': self.device,
            'random_state': self.random_state
        }
        
        if self.model is not None and isinstance(self.model, nn.Module):
            state['model_state_dict'] = self.model.state_dict()
            state['model_config'] = self._get_model_config()
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set model state from saved dictionary.
        
        Parameters
        ----------
        state : dict
            Model state dictionary
        """
        self.is_fitted = state['is_fitted']
        self.device = state.get('device', 'cpu')
        self.random_state = state.get('random_state')
        
        if 'model_state_dict' in state:
            # Recreate model architecture
            self._create_model_from_config(state.get('model_config', {}))
            # Load weights
            self.model.load_state_dict(state['model_state_dict'])
            self.model.to(self.device)
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for recreation."""
        return {}
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> None:
        """Create model from configuration."""
        pass