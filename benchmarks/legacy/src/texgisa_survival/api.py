# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
High-level API for TexGISa Survival Analysis

This module provides a simplified, user-friendly interface for the TexGISa
(Time-dependent EXtreme Gradient Integration for Survival Analysis) model.
"""

from typing import Optional, List, Dict, Union, Tuple, Any
import pandas as pd
import numpy as np
import torch
import warnings
from pathlib import Path
import joblib

from .models import TEXGISAModel


class TexGISa:
    """
    TexGISa: Time-dependent EXtreme Gradient Integration for Survival Analysis.

    A deep learning model for survival analysis that provides interpretable
    predictions through time-dependent feature importance and supports
    integration of expert domain knowledge.

    The network predicts one hazard per time interval; a Cox risk head on the
    shared representation gives the risk score and shifts every interval's
    hazard (proportional-hazards coupling). The hazards are trained with the
    piecewise-constant-hazard likelihood and calibrated on validation data
    after training. Defaults are the values most often selected by nested
    cross-validation in the package benchmark (``benchmarks/``).

    Parameters
    ----------
    hidden_layers : list of int, default [256, 128]
        'mlp': hidden layer widths. 'resnet': width hidden_layers[0] and
        len(hidden_layers) residual blocks
    num_time_bins : int, default 20
        Number of discrete time intervals for hazard prediction
    dropout : float, default 0.2
        Dropout probability for regularization
    activation : str, default 'selu'
        Activation function ('relu', 'elu', 'selu', 'gelu', 'tanh', ...)
    lambda_smooth : float, default 0.0
        Weight for temporal smoothness regularization
    lambda_expert : float, default 0.0
        Weight for expert knowledge constraints
    ig_steps : int, default 50
        Number of integration steps for TEXGI importance calculation
    time_binning : {'quantile', 'uniform'}, default 'quantile'
        Edges at quantiles of the event times, or equal-width intervals
    risk_score : {'expected_time', 'hazard_sum'}, default 'expected_time'
        Risk score when lambda_cox=0 (no Cox head): minus the restricted mean
        survival time, or the sum of interval hazards
    weight_decay : float, default 0.01
        Decoupled weight decay (AdamW)
    architecture : {'resnet', 'mlp'}, default 'resnet'
        Residual MLP or plain MLP backbone
    **kwargs
        Further model options of
        :class:`~texgisa_survival.models.TEXGISAModel`: ``norm`` ('layer'),
        ``lambda_cox`` (1.0), ``likelihood`` ('pch'), ``calibrate`` (True),
        ``ema_decay`` (0.9), ``input_noise`` (0.1), ``input_gate`` (False),
        ``es_metric`` ('loss').
    random_state : int, optional
        Random seed for reproducibility
    device : str, optional
        Device for computation ('cpu', 'cuda', 'mps'). Auto-detected if None.
    verbose : int, default 1
        Verbosity level (0=silent, 1=progress, 2=debug)

    Examples
    --------
    >>> from texgisa_survival import TexGISa
    >>>
    >>> # Basic usage
    >>> model = TexGISa(random_state=42)
    >>> model.fit(X_train, time_train, event_train)  # holds out 20% for validation
    >>> predictions = model.predict_survival(X_test, times=[12, 24, 36])
    >>>
    >>> # With expert knowledge
    >>> model = TexGISa(lambda_expert=0.1)
    >>> model.add_expert_rule('age', '>=', 'mean', sign='+1')
    >>> model.add_expert_rule('biomarker', '>', 0.5, sign='-1')
    >>> model.fit(X_train, time_train, event_train)
    >>> importance = model.get_feature_importance(method='texgi')
    """

    def __init__(
        self,
        hidden_layers: Optional[List[int]] = None,
        num_time_bins: int = 20,
        dropout: float = 0.2,
        activation: str = "selu",
        lambda_smooth: float = 0.0,
        lambda_expert: float = 0.0,
        ig_steps: int = 50,
        time_binning: str = "quantile",
        risk_score: str = "expected_time",
        weight_decay: float = 0.01,
        architecture: str = "resnet",
        random_state: Optional[int] = None,
        device: Optional[str] = None,
        verbose: int = 1,
        **kwargs,
    ):
        """Initialize TexGISa model."""

        self.random_state = random_state
        self.verbose = verbose
        self.is_fitted = False
        self.feature_names = None
        self.expert_rules = []

        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.verbose:
            print(f"Initialized TexGISa model on {self.device}")

        # Initialize underlying model
        self.model = TEXGISAModel(
            hidden_layers=hidden_layers,
            num_time_bins=num_time_bins,
            dropout=dropout,
            activation=activation,
            lambda_smooth=lambda_smooth,
            lambda_expert=lambda_expert,
            ig_steps=ig_steps,
            time_binning=time_binning,
            risk_score=risk_score,
            weight_decay=weight_decay,
            architecture=architecture,
            device=self.device,
            random_state=random_state,
            **kwargs,
        )

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        time: Union[pd.Series, np.ndarray],
        event: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple] = None,
        epochs: int = 300,
        batch_size: int = 512,
        learning_rate: float = 0.003,
        early_stopping: bool = True,
        patience: int = 15,
        **kwargs,
    ) -> "TexGISa":
        """
        Train the TexGISa model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        time : array-like of shape (n_samples,)
            Time to event or censoring
        event : array-like of shape (n_samples,)
            Event indicator (1=event, 0=censored)
        validation_data : tuple, optional
            (X_val, time_val, event_val) for early stopping, checkpointing and
            calibration. If None, ``validation_split`` (default 0.2) of the
            training data is held out at random.
        epochs : int, default 300
            Maximum number of training epochs
        batch_size : int, default 512
            Batch size for training
        learning_rate : float, default 0.003
            Learning rate
        early_stopping : bool, default True
            Whether to use early stopping
        patience : int, default 15
            Patience for early stopping
        **kwargs : dict
            Additional training parameters (e.g. ``validation_split``)

        Returns
        -------
        self : TexGISa
            Fitted model instance
        """
        # Convert to numpy if needed and store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if isinstance(time, pd.Series):
            time = time.values
        if isinstance(event, pd.Series):
            event = event.values

        # Validate inputs
        self._validate_inputs(X, time, event)

        # Pass expert rules to model if any
        if self.expert_rules:
            kwargs["expert_rules"] = {"rules": self.expert_rules}

        # Handle verbose parameter - allow override from kwargs
        verbose = kwargs.pop("verbose", self.verbose)

        # Train the model
        if verbose:
            print(f"Training TexGISa model...")

        self.model.fit(
            X,
            time,
            event,
            feature_names=self.feature_names,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            patience=patience,
            verbose=verbose,
            **kwargs,
        )

        self.is_fitted = True

        if self.verbose:
            print("Training completed!")

        return self

    def predict_survival(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        times: Optional[Union[List[float], np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Predict survival probabilities at specified times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        times : array-like of shape (n_times,), optional
            Times at which to evaluate survival. If None, uses training time bins.

        Returns
        -------
        survival_probs : np.ndarray of shape (n_samples, n_times)
            Survival probabilities S(t) for each sample at each time point
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_survival(X, times)

    def predict_risk(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict risk scores (higher = higher risk of event).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix

        Returns
        -------
        risk_scores : np.ndarray of shape (n_samples,)
            Risk scores for each sample
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        return self.model.predict_risk(X)

    def add_expert_rule(
        self,
        feature: Union[str, int],
        relation: str,
        threshold: Union[str, float],
        sign: str = "+1",
        weight: float = 1.0,
        min_magnitude: float = 0.01,
    ) -> None:
        """
        Add expert knowledge rule for feature importance constraints.

        Expert rules help guide the model's feature importance learning by
        specifying expected relationships between features and survival risk.

        Parameters
        ----------
        feature : str or int
            Feature name or index
        relation : str
            Relation operator ('>', '<', '>=', '<=', '==')
        threshold : str or float
            Threshold value or 'mean', 'median'
        sign : str, default '+1'
            Expected sign of effect on risk:
            - '+1': higher feature values increase risk
            - '-1': higher feature values decrease risk
        weight : float, default 1.0
            Importance weight of this rule (higher = stronger constraint)
        min_magnitude : float, default 0.01
            Minimum magnitude for the effect

        Examples
        --------
        >>> model.add_expert_rule('age', '>=', 'mean', sign='+1', weight=1.5)
        >>> model.add_expert_rule('treatment_A', '==', 1, sign='-1', weight=2.0)
        >>> model.add_expert_rule('biomarker', '>', 0.5, sign='+1')
        """
        # Convert sign to integer
        if isinstance(sign, str):
            sign_int = 1 if sign == "+1" else -1
        else:
            sign_int = 1 if sign > 0 else -1

        rule = {
            "feature": feature,
            "relation": relation,
            "threshold": threshold,
            "sign": sign_int,
            "weight": weight,
            "min_magnitude": min_magnitude,
        }
        self.expert_rules.append(rule)

        if self.verbose:
            print(f"Added expert rule: {feature} {relation} {threshold} -> {sign}")

    def get_feature_importance(
        self, method: str = "texgi", n_steps: Optional[int] = None, **kwargs
    ) -> pd.DataFrame:
        """
        Get feature importance scores.

        Parameters
        ----------
        method : str, default 'texgi'
            Method for importance calculation:
            - 'texgi': Time-dependent extreme gradient integration (recommended)
            - 'permutation': Permutation importance
        n_steps : int, optional
            Number of integration steps for TEXGI. If None, uses model default.

        Returns
        -------
        importance : pd.DataFrame
            DataFrame with columns ['feature', 'importance', 'std'] sorted by
            importance (descending)
        """
        self._check_is_fitted()

        importance_dict = self.model.get_feature_importance(
            method=method, n_steps=n_steps, **kwargs
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": importance_dict["importance"],
                "std": importance_dict.get("std", np.zeros(len(self.feature_names))),
            }
        )

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        n_steps: Optional[int] = None,
        time_bins: Optional[int] = None,
        baseline_quantile: float = 0.98,
    ) -> np.ndarray:
        """
        Time-resolved TEXGI attributions for individual samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain
        n_steps : int, optional
            Integration steps; defaults to the model's ``ig_steps``
        time_bins : int, optional
            Number of time bins to subsample (evenly spaced); all bins if None
        baseline_quantile : float, default 0.98
            Upper-tail quantile of the training data used as the extreme baseline

        Returns
        -------
        phi : np.ndarray of shape (n_bins, n_samples, n_features)
            Attribution of each feature to the hazard of each sample in each bin
        """
        from .texgi_utils import texgi_time_series

        self._check_is_fitted()
        if isinstance(X, pd.DataFrame):
            X = X.values
        inner = self.model
        X_std = torch.FloatTensor(inner.scaler.transform(np.asarray(X, dtype=np.float32))).to(
            self.device
        )
        ref = inner.training_data_sample
        baseline = torch.quantile(ref, baseline_quantile, dim=0, keepdim=True).repeat(
            X_std.shape[0], 1
        )

        inner.model.eval()
        phi = texgi_time_series(
            inner.model, X_std, X_baseline=baseline, M=n_steps or inner.ig_steps, t_sample=time_bins
        )
        return phi.detach().cpu().numpy()

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        time: Union[pd.Series, np.ndarray],
        event: Union[pd.Series, np.ndarray],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        time : array-like of shape (n_samples,)
            Time to event or censoring
        event : array-like of shape (n_samples,)
            Event indicator
        metrics : list of str, optional
            Metrics to compute. Default: ['c-index', 'brier_score']
            Available: 'c-index', 'brier_score', 'ibs'

        Returns
        -------
        scores : dict
            Dictionary of metric scores
        """
        self._check_is_fitted()

        if metrics is None:
            metrics = ["c-index", "brier_score"]

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(time, pd.Series):
            time = time.values
        if isinstance(event, pd.Series):
            event = event.values

        scores = {}

        if "c-index" in metrics:
            from .metrics import concordance_index

            risk_scores = self.predict_risk(X)
            scores["c-index"] = concordance_index(time, risk_scores, event)

        if "brier_score" in metrics:
            from .metrics import brier_score

            times = np.percentile(time[event == 1], [25, 50, 75])
            survival_probs = self.predict_survival(X, times)
            scores["brier_score"] = float(
                np.mean(brier_score(time, survival_probs, event, eval_times=times))
            )

        if "ibs" in metrics:
            from .metrics import integrated_brier_score

            times = np.linspace(time[event == 1].min(), time[event == 1].max(), 20)
            survival_probs = self.predict_survival(X, times)
            scores["ibs"] = integrated_brier_score(time, survival_probs, event, eval_times=times)

        return scores

    def plot_survival_curves(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        times: Optional[np.ndarray] = None,
        sample_indices: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        title: str = "TexGISa Predicted Survival Curves",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot survival curves for selected samples.

        Parameters
        ----------
        X : array-like
            Feature matrix
        times : array-like, optional
            Time points for evaluation
        sample_indices : list of int, optional
            Indices of samples to plot. If None, plots first 5 samples.
        labels : list of str, optional
            Labels for each curve
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        from .visualization import plot_survival_curves

        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        survival_probs = self.predict_survival(X, times)

        plot_survival_curves(
            survival_probs=survival_probs,
            times=times,
            sample_indices=sample_indices,
            labels=labels,
            title=title,
            save_path=save_path,
        )

    def save(self, filepath: str) -> None:
        """
        Save trained model to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model (e.g., 'model.pkl')
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Get model state
        save_dict = {
            "model_state": self.model.get_state(),
            "feature_names": self.feature_names,
            "expert_rules": self.expert_rules,
            "device": self.device,
            "version": __import__("texgisa_survival").__version__,
        }

        joblib.dump(save_dict, filepath)

        if self.verbose:
            print(f"Model saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load trained model from disk.

        Parameters
        ----------
        filepath : str
            Path to the saved model file
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        save_dict = joblib.load(filepath)

        # Restore model state
        self.feature_names = save_dict["feature_names"]
        self.expert_rules = save_dict["expert_rules"]
        self.device = save_dict.get("device", "cpu")

        self.model.set_state(save_dict["model_state"])
        self.is_fitted = True

        if self.verbose:
            print(f"Model loaded from {filepath}")

    def _validate_inputs(self, X, time, event):
        """Validate input data."""
        if len(X) != len(time) or len(X) != len(event):
            raise ValueError("X, time, and event must have the same length")

        if np.any(time < 0):
            raise ValueError("Time values must be non-negative")

        if not np.all(np.isin(event, [0, 1])):
            raise ValueError("Event values must be 0 or 1")

        if np.any(np.isnan(X)) or np.any(np.isnan(time)) or np.any(np.isnan(event)):
            raise ValueError("Input contains NaN values")

    def _check_is_fitted(self):
        """Check if model has been fitted."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")


def load_model(filepath: str, verbose: int = 1) -> TexGISa:
    """
    Load a saved TexGISa model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model file
    verbose : int, default 1
        Verbosity level

    Returns
    -------
    model : TexGISa
        Loaded model instance

    Examples
    --------
    >>> model = load_model('trained_model.pkl')
    >>> predictions = model.predict_risk(X_test)
    """
    model = TexGISa(verbose=verbose)
    model.load(filepath)
    return model
