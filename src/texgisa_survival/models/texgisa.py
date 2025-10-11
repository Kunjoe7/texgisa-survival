# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
TEXGISA (Time-dependent EXtreme Gradient Integration for Survival Analysis) model implementation.

This is the main innovation of the DHAI Lab - an interpretable survival model
that incorporates expert knowledge and provides time-dependent feature importance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseSurvivalModel


class MultiTaskSurvivalNet(nn.Module):
    """
    Multi-task neural network for survival analysis with discrete time intervals.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_layers: List[int] = None,
                 num_time_bins: int = 20,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                getattr(nn, activation.upper() if hasattr(nn, activation.upper()) else 'ReLU')(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer for hazards at each time bin
        layers.append(nn.Linear(prev_dim, num_time_bins))
        layers.append(nn.Sigmoid())  # Hazards in [0,1]
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class AdversarialGenerator(nn.Module):
    """
    Generator for creating adversarial extreme baselines for integrated gradients.
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 16,
                 extreme_dim: int = 1,
                 hidden_dim: int = 128):
        super().__init__()
        
        total_input = input_dim + latent_dim + extreme_dim
        
        self.network = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, z, e):
        """
        Args:
            x: Input features [batch_size, input_dim]
            z: Latent noise [batch_size, latent_dim]  
            e: Extreme code [batch_size, extreme_dim]
        """
        combined = torch.cat([x, z, e], dim=1)
        return self.network(combined)


class TEXGISAModel(BaseSurvivalModel):
    """
    TEXGISA: Time-dependent EXtreme Gradient Integration for Survival Analysis.
    
    This model provides interpretable survival analysis by:
    1. Using discrete time intervals for hazard prediction
    2. Incorporating expert knowledge through rule-based constraints
    3. Computing time-dependent feature importance via TEXGI
    4. Training adversarial baselines for better attribution
    
    Parameters
    ----------
    hidden_layers : list of int, default [128, 64, 32]
        Hidden layer dimensions
    num_time_bins : int, default 20
        Number of discrete time intervals
    dropout : float, default 0.1
        Dropout probability
    activation : str, default 'relu'
        Activation function
    lambda_smooth : float, default 0.0
        Weight for temporal smoothness regularization
    lambda_expert : float, default 0.0
        Weight for expert knowledge constraints
    ig_steps : int, default 20
        Number of integration steps for TEXGI
    device : str, default 'cpu'
        Device for computation
    random_state : int, optional
        Random seed
    """
    
    def __init__(self,
                 hidden_layers: List[int] = None,
                 num_time_bins: int = 20,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 lambda_smooth: float = 0.0,
                 lambda_expert: float = 0.0,
                 ig_steps: int = 20,
                 device: str = 'cpu',
                 random_state: Optional[int] = None,
                 **kwargs):
        super().__init__(device=device, random_state=random_state)
        
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.num_time_bins = num_time_bins
        self.dropout = dropout
        self.activation = activation
        self.lambda_smooth = lambda_smooth
        self.lambda_expert = lambda_expert
        self.ig_steps = ig_steps
        
        self.scaler = StandardScaler()
        self.time_bins = None
        self.generator = None
        self.expert_rules = []
        self.feature_names = None
        
    def add_expert_rule(self,
                       feature: str,
                       relation: str,
                       threshold: str,
                       sign: int = 1,
                       weight: float = 1.0):
        """Add expert knowledge rule for feature importance constraints."""
        rule = {
            'feature': feature,
            'relation': relation,
            'threshold': threshold,
            'sign': sign,
            'weight': weight
        }
        self.expert_rules.append(rule)
        
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
            expert_rules: Optional[Dict] = None,
            **kwargs) -> None:
        """
        Fit the TEXGISA model.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        time : np.ndarray
            Survival times
        event : np.ndarray
            Event indicators
        validation_data : tuple, optional
            Validation data
        epochs : int
            Training epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        early_stopping : bool
            Whether to use early stopping
        patience : int
            Early stopping patience
        verbose : int
            Verbosity level
        expert_rules : dict, optional
            Expert rules dictionary
        **kwargs
            Additional parameters
        """
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        # Normalize features
        X = self.scaler.fit_transform(X.astype(np.float32))
        
        # Create time discretization
        self.time_bins = np.linspace(time.min(), time.max(), self.num_time_bins + 1)
        
        # Discretize survival times
        time_discrete = np.digitize(time, self.time_bins[1:])
        time_discrete = np.clip(time_discrete, 0, self.num_time_bins - 1)
        
        # Create supervision labels and masks
        labels, masks = self._create_supervision(time_discrete, event)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        masks_tensor = torch.FloatTensor(masks).to(self.device)
        
        # Create model
        input_dim = X.shape[1]
        self.model = MultiTaskSurvivalNet(
            input_dim=input_dim,
            hidden_layers=self.hidden_layers,
            num_time_bins=self.num_time_bins,
            dropout=self.dropout,
            activation=self.activation
        ).to(self.device)
        
        # Create adversarial generator for TEXGI
        self._train_generator(X_tensor, verbose=verbose)
        
        # Set up expert rules
        if expert_rules and 'rules' in expert_rules:
            self.expert_rules = expert_rules['rules']
            
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience//2, factor=0.5
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            
            # Mini-batch training
            n_samples = X_tensor.shape[0]
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                X_batch = X_tensor[batch_indices]
                labels_batch = labels_tensor[batch_indices]
                masks_batch = masks_tensor[batch_indices]
                
                optimizer.zero_grad()
                
                # Forward pass
                hazards = self.model(X_batch)
                
                # Main loss: masked BCE
                main_loss = self._masked_bce_loss(hazards, labels_batch, masks_batch)
                
                # Regularization losses
                smooth_loss = self._temporal_smoothness_loss(hazards)
                expert_loss = torch.tensor(0.0, device=self.device)
                
                if self.lambda_expert > 0 and self.expert_rules:
                    expert_loss = self._compute_expert_loss(X_batch, hazards)
                
                total_loss = main_loss + self.lambda_smooth * smooth_loss + self.lambda_expert * expert_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            epoch_loss /= (n_samples // batch_size + 1)
            
            # Validation
            if validation_data is not None:
                val_loss = self._validate(validation_data)
                scheduler.step(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")
                    
                if early_stopping and patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={epoch_loss:.4f}")

        self.is_fitted = True

        # Initialize TEXGI explainer with training data sample
        if verbose:
            print("Initializing TEXGI explainer...")

        background_size = min(100, len(X_tensor))
        background_indices = torch.randperm(len(X_tensor))[:background_size]
        background_data = X_tensor[background_indices].detach()

        from ..explainers.texgi import TEXGIExplainer
        self.texgi_explainer = TEXGIExplainer(
            background_data=background_data,
            batch_size=min(16, background_size),
            random_alpha=True,
            k=max(1, self.ig_steps // 5),  # Use ig_steps/5 reference samples
            scale_by_inputs=True,
            device=self.device
        )

        # Save a sample of training data for importance computation
        self.training_data_sample = X_tensor[:min(200, len(X_tensor))].detach()

        if verbose:
            print("TEXGI explainer initialized")

    def predict_survival(self,
                        X: np.ndarray,
                        times: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Preprocess
        if hasattr(X, 'values'):
            X = X.values
        X = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Predict hazards
        self.model.eval()
        with torch.no_grad():
            hazards = self.model(X_tensor).cpu().numpy()
        
        # Convert hazards to survival probabilities
        survival_probs = self._hazards_to_survival(hazards)
        
        if times is None:
            return survival_probs
        
        # Interpolate to requested times
        return self._interpolate_survival(survival_probs, times)
        
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Use sum of hazards as risk score
        hazards = self._predict_hazards(X)
        return hazards.sum(axis=1)
        
    def get_feature_importance(self,
                             method: str = 'texgi',
                             n_steps: int = None,
                             **kwargs) -> Dict[str, np.ndarray]:
        """
        Compute feature importance using TEXGI or other methods.
        
        Parameters
        ----------
        method : str
            Method to use ('texgi', 'weights', 'permutation')
        n_steps : int, optional
            Number of integration steps for TEXGI
        **kwargs
            Additional parameters
            
        Returns
        -------
        importance : dict
            Feature importance scores
        """
        if method == 'texgi' or method == 'default':
            return self._compute_texgi_importance(n_steps or self.ig_steps)
        elif method == 'weights':
            return self._get_weight_importance()
        elif method == 'permutation':
            return self._get_permutation_importance(**kwargs)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _create_supervision(self, time_discrete, event):
        """Create supervision labels and masks for discrete time bins."""
        n_samples = len(time_discrete)
        labels = np.zeros((n_samples, self.num_time_bins))
        masks = np.zeros((n_samples, self.num_time_bins))
        
        # Convert to numpy array if it's a pandas Series
        if hasattr(event, 'values'):
            event = event.values
        
        for i in range(n_samples):
            t_idx = time_discrete[i]
            # Mask all time bins up to and including the event/censoring time
            masks[i, :t_idx+1] = 1
            # If event occurred, set hazard to 1 at that time
            if event[i] == 1:
                labels[i, t_idx] = 1
                
        return labels, masks
    
    def _masked_bce_loss(self, hazards, labels, masks):
        """Compute masked binary cross entropy loss."""
        eps = 1e-7
        hazards = torch.clamp(hazards, eps, 1-eps)
        bce = -(labels * torch.log(hazards) + (1 - labels) * torch.log(1 - hazards))
        masked_loss = (bce * masks).sum() / masks.sum().clamp_min(1.0)
        return masked_loss
    
    def _temporal_smoothness_loss(self, hazards):
        """Compute temporal smoothness regularization."""
        if self.lambda_smooth <= 0:
            return torch.tensor(0.0, device=hazards.device)
        diff = hazards[:, 1:] - hazards[:, :-1]
        return F.smooth_l1_loss(diff, torch.zeros_like(diff))
    
    def _compute_expert_loss(self, X, hazards):
        """Compute expert knowledge constraint loss using TEXGI attributions."""
        if not self.expert_rules:
            return torch.tensor(0.0, device=X.device)

        if not hasattr(self, 'texgi_explainer'):
            # During early training, explainer might not be ready
            return torch.tensor(0.0, device=X.device)

        try:
            # Create model wrapper for TEXGI
            class ModelWrapper(nn.Module):
                def __init__(self, model, num_time_bins):
                    super().__init__()
                    self.model = model
                    self.num_time_bins = num_time_bins

                def forward(self, x):
                    output = self.model(x)
                    return [output[:, i:i+1] for i in range(self.num_time_bins)]

            wrapped_model = ModelWrapper(self.model, self.num_time_bins)

            # Compute attributions for current batch
            self.model.eval()
            with torch.no_grad():
                attributions_list = self.texgi_explainer.compute_attributions(
                    model=wrapped_model,
                    input_tensor=X,
                    sparse_labels=None
                )
            self.model.train()

            # Average across time bins
            attributions = sum(attributions_list) / len(attributions_list)  # [batch_size, n_features]

        except Exception as e:
            # If TEXGI computation fails, return zero loss
            print(f"Warning: Expert loss computation failed: {e}")
            return torch.tensor(0.0, device=X.device)

        expert_loss = torch.tensor(0.0, device=X.device)

        for rule in self.expert_rules:
            try:
                # Get feature index
                feature_idx = self._get_feature_index(rule['feature'])

                feature_attr = attributions[:, feature_idx]  # [batch_size]
                feature_values = X[:, feature_idx]

                # Compute threshold
                threshold = self._compute_threshold(rule['threshold'], X, feature_idx)

                # Create mask for samples satisfying the condition
                relation = rule['relation']
                if relation == '>':
                    mask = feature_values > threshold
                elif relation == '>=':
                    mask = feature_values >= threshold
                elif relation == '<':
                    mask = feature_values < threshold
                elif relation == '<=':
                    mask = feature_values <= threshold
                elif relation == '==':
                    mask = torch.abs(feature_values - threshold) < 1e-5
                else:
                    continue

                # For samples satisfying condition, check if attribution matches expected sign
                if mask.any():
                    expected_sign = rule['sign']  # +1 or -1
                    actual_attr = feature_attr[mask]

                    # Penalize when attribution sign doesn't match expectation
                    # Loss is high when attribution has opposite sign
                    sign_penalty = torch.where(
                        actual_attr * expected_sign < 0,  # Wrong sign
                        torch.abs(actual_attr),
                        torch.zeros_like(actual_attr)
                    )

                    # Weight by rule importance
                    rule_loss = (sign_penalty * rule['weight']).mean()
                    expert_loss += rule_loss

            except Exception as e:
                # Skip problematic rules
                continue

        return expert_loss
    
    def _train_generator(self, X_tensor, epochs=100, verbose=False):
        """Train adversarial generator for extreme baselines."""
        if self.generator is None:
            self.generator = AdversarialGenerator(
                input_dim=X_tensor.shape[1]
            ).to(self.device)
        
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        
        for epoch in range(epochs):
            # Sample noise and extreme codes
            batch_size = min(128, X_tensor.shape[0])
            indices = torch.randperm(X_tensor.shape[0])[:batch_size]
            X_batch = X_tensor[indices]
            
            z = torch.randn(batch_size, 16, device=self.device)
            e = torch.rand(batch_size, 1, device=self.device)  # Simplified extreme sampling
            
            # Generate adversarial examples
            X_adv = self.generator(X_batch, z, e)
            
            # Compute loss (maximize risk, minimize distance)
            with torch.no_grad():
                hazards = self.model(X_adv) if self.model else torch.zeros(batch_size, self.num_time_bins, device=self.device)
                risk = hazards.sum(dim=1)
            
            distance = ((X_adv - X_batch) ** 2).sum(dim=1)
            loss = (-risk + 0.1 * distance).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if verbose:
            print("Generator training completed")
    
    def _compute_texgi_importance(self, n_steps):
        """Compute TEXGI feature importance using Expected Gradients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing feature importance")

        if not hasattr(self, 'texgi_explainer'):
            raise ValueError("TEXGI explainer not initialized. Model may not have been trained properly.")

        # Use stored training data sample
        X_sample = self.training_data_sample.to(self.device)

        # Create a wrapper that converts model output to list format expected by TEXGI
        class ModelWrapper(nn.Module):
            def __init__(self, model, num_time_bins):
                super().__init__()
                self.model = model
                self.num_time_bins = num_time_bins

            def forward(self, x):
                # Model outputs [batch_size, num_time_bins]
                # TEXGI expects list of [batch_size, 1] for each time bin
                output = self.model(x)
                return [output[:, i:i+1] for i in range(self.num_time_bins)]

        wrapped_model = ModelWrapper(self.model, self.num_time_bins)
        wrapped_model.eval()

        # Compute TEXGI attributions
        with torch.no_grad():
            attributions_list = self.texgi_explainer.compute_attributions(
                model=wrapped_model,
                input_tensor=X_sample,
                sparse_labels=None
            )

        # Average across time bins
        avg_attributions = sum(attributions_list) / len(attributions_list)

        # Convert to numpy and compute statistics
        avg_attributions_np = avg_attributions.cpu().numpy()

        # Compute mean and std across samples
        importance = np.abs(avg_attributions_np).mean(axis=0)
        std = np.abs(avg_attributions_np).std(axis=0)

        # Normalize to sum to 1
        importance = importance / (importance.sum() + 1e-10)

        return {
            'importance': importance,
            'std': std
        }
    
    def _hazards_to_survival(self, hazards):
        """Convert discrete hazards to survival probabilities."""
        # S(t) = ∏(1 - h(s)) for s ≤ t
        survival = np.cumprod(1 - hazards, axis=1)
        return survival
    
    def _predict_hazards(self, X):
        """Predict hazards for input data."""
        if hasattr(X, 'values'):
            X = X.values
        X = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            hazards = self.model(X_tensor).cpu().numpy()
        
        return hazards
    
    def _interpolate_survival(self, survival_probs, times):
        """Interpolate survival probabilities to requested times."""
        # Simple linear interpolation between time bins
        time_points = self.time_bins[1:]  # Use bin centers
        
        result = np.zeros((survival_probs.shape[0], len(times)))
        for i, t in enumerate(times):
            if t <= time_points[0]:
                result[:, i] = 1.0
            elif t >= time_points[-1]:
                result[:, i] = survival_probs[:, -1]
            else:
                # Linear interpolation
                idx = np.searchsorted(time_points, t)
                t0, t1 = time_points[idx-1], time_points[idx]
                w = (t - t0) / (t1 - t0)
                result[:, i] = (1-w) * survival_probs[:, idx-1] + w * survival_probs[:, idx]
        
        return result
    
    def _validate(self, validation_data):
        """Compute validation loss."""
        X_val, time_val, event_val = validation_data
        
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        X_val = self.scaler.transform(X_val.astype(np.float32))
        
        # Discretize and create supervision
        time_discrete = np.digitize(time_val, self.time_bins[1:])
        time_discrete = np.clip(time_discrete, 0, self.num_time_bins - 1)
        labels, masks = self._create_supervision(time_discrete, event_val)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_val).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        masks_tensor = torch.FloatTensor(masks).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            hazards = self.model(X_tensor)
            loss = self._masked_bce_loss(hazards, labels_tensor, masks_tensor)
        
        return loss.item()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        return {
            'hidden_layers': self.hidden_layers,
            'num_time_bins': self.num_time_bins,
            'dropout': self.dropout,
            'activation': self.activation,
            'lambda_smooth': self.lambda_smooth,
            'lambda_expert': self.lambda_expert,
            'ig_steps': self.ig_steps,
            'expert_rules': self.expert_rules,
            'feature_names': self.feature_names,
            'time_bins': self.time_bins.tolist() if self.time_bins is not None else None
        }
    
    def _create_model_from_config(self, config: Dict[str, Any]) -> None:
        """Create model from configuration."""
        input_dim = len(config.get('feature_names', []))
        if input_dim > 0:
            self.model = MultiTaskSurvivalNet(
                input_dim=input_dim,
                hidden_layers=config.get('hidden_layers', self.hidden_layers),
                num_time_bins=config.get('num_time_bins', self.num_time_bins),
                dropout=config.get('dropout', self.dropout),
                activation=config.get('activation', self.activation)
            ).to(self.device)

            self.generator = AdversarialGenerator(input_dim).to(self.device)

            # Restore other attributes
            self.expert_rules = config.get('expert_rules', [])
            self.feature_names = config.get('feature_names', [])
            if 'time_bins' in config and config['time_bins']:
                self.time_bins = np.array(config['time_bins'])

    def _get_feature_index(self, feature):
        """Get feature index from name or integer index."""
        if isinstance(feature, int):
            if feature < 0 or (self.feature_names and feature >= len(self.feature_names)):
                raise ValueError(f"Feature index {feature} out of range")
            return feature
        elif isinstance(feature, str):
            if not self.feature_names:
                raise ValueError("Feature names not available")
            try:
                return self.feature_names.index(feature)
            except ValueError:
                raise ValueError(f"Feature '{feature}' not found in feature names")
        else:
            raise TypeError(f"Feature must be int or str, got {type(feature)}")

    def _compute_threshold(self, threshold, X, feature_idx):
        """Compute threshold value from specification."""
        if isinstance(threshold, (int, float)):
            return float(threshold)
        elif threshold == 'mean':
            return X[:, feature_idx].mean().item()
        elif threshold == 'median':
            return X[:, feature_idx].median().item()
        elif threshold == 'min':
            return X[:, feature_idx].min().item()
        elif threshold == 'max':
            return X[:, feature_idx].max().item()
        else:
            raise ValueError(f"Unknown threshold specification: {threshold}")