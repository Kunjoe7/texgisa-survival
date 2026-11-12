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

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


def _activation(name: str):
    if name.lower() not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'; choose from {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name.lower()]


def _norm(kind: str, dim: int) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    if kind == "layer":
        return nn.LayerNorm(dim)
    raise ValueError("norm must be 'batch' or 'layer'")


class MultiTaskSurvivalNet(nn.Module):
    """
    Multi-task neural network for survival analysis with discrete time intervals.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = None,
        num_time_bins: int = 20,
        dropout: float = 0.1,
        activation: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        act = _activation(activation)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    _norm(norm, hidden_dim),
                    act(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer for hazards at each time bin
        layers.append(nn.Linear(prev_dim, num_time_bins))
        layers.append(nn.Sigmoid())  # Hazards in [0,1]

        self.network = nn.Sequential(*layers)
        self.feature_dim = prev_dim

    def forward(self, x):
        return self.network(x)

    def features(self, x):
        """Shared representation before the hazard layer."""
        return self.network[:-2](x)

    def logits(self, f):
        """Hazard logits from the shared representation."""
        return self.network[-2](f)


class _FeatureGate(nn.Module):
    """Input feature gate: an MLP compresses then restores the inputs and re-weights them."""

    def __init__(self, dim: int, reduction: int = 2):
        super().__init__()
        hidden = max(4, dim // reduction)
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.gate(x)


class SurvivalModule(nn.Module):
    """
    Hazard network of TexGISa. A backbone maps the (optionally gated) inputs to a
    shared representation f. A hazard head gives one logit z_k per interval and,
    if `cox_head` is set, a Cox risk head gives a scalar risk r(f) that is added
    to every logit (proportional-hazards coupling). The interval hazard is
    h_k = 1 - exp(-softplus(z_k / T + b)), i.e. softplus(.) is the cumulative
    hazard of interval k; T and b are a temperature and a bias fitted after
    training (calibration). forward(x) returns the hazards, so attribution
    code works on the hazards directly.
    """

    def __init__(
        self,
        base: nn.Module,
        cox_head: bool = False,
        input_gate: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.gate = input_gate
        self.base = base
        self.cox = nn.Linear(base.feature_dim, 1) if cox_head else None
        self.register_buffer("temperature", torch.ones(()))
        self.register_buffer("logit_bias", torch.zeros(()))

    def _features(self, x):
        if self.gate is not None:
            x = self.gate(x)
        return self.base.features(x)

    def _logits(self, f):
        z = self.base.logits(f)
        if self.cox is not None:
            z = z + self.cox(f)  # the Cox risk shifts every interval's logit
        return z

    def _to_hazard(self, z):
        return _softplus_hazard(z / self.temperature + self.logit_bias)

    def hazards_and_risk(self, x):
        f = self._features(x)
        h = self._to_hazard(self._logits(f))
        r = self.cox(f).squeeze(1) if self.cox is not None else None
        return h, r

    def raw_logits(self, x):
        return self._logits(self._features(x))

    def risk(self, x):
        return self.cox(self._features(x)).squeeze(1)

    def forward(self, x):
        return self.hazards_and_risk(x)[0]


def _softplus_hazard(z):
    """Interval hazard whose cumulative hazard is softplus(z)."""
    return 1 - torch.exp(-F.softplus(z))


class ResNetSurvivalNet(nn.Module):
    """
    Residual MLP for tabular data (Gorishniy et al., 2021): a linear stem, blocks
    x + Dropout(Linear(Dropout(act(Linear(Norm(x)))))), then Norm, act and a
    linear hazard head.
    """

    def __init__(
        self,
        input_dim: int,
        num_time_bins: int,
        width: int = 128,
        n_blocks: int = 2,
        dropout: float = 0.1,
        norm: str = "batch",
        activation: str = "relu",
    ):
        super().__init__()
        act = _activation(activation)
        self.stem = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                _norm(norm, width),
                nn.Linear(width, 2 * width),
                act(),
                nn.Dropout(dropout),
                nn.Linear(2 * width, width),
                nn.Dropout(dropout),
            )
            for _ in range(n_blocks)
        )
        self.out = nn.Sequential(_norm(norm, width), act())
        self.head = nn.Linear(width, num_time_bins)
        self.feature_dim = width

    def features(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = x + block(x)
        return self.out(x)

    def logits(self, f):
        return self.head(f)

    def forward(self, x):
        return torch.sigmoid(self.logits(self.features(x)))


class AdversarialGenerator(nn.Module):
    """
    Generator for creating adversarial extreme baselines for integrated gradients.
    """

    def __init__(
        self, input_dim: int, latent_dim: int = 16, extreme_dim: int = 1, hidden_dim: int = 128
    ):
        super().__init__()

        total_input = input_dim + latent_dim + extreme_dim

        self.network = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
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


# Hyperparameters stored with a saved model
_HYPERPARAMETERS = (
    "hidden_layers",
    "num_time_bins",
    "dropout",
    "activation",
    "lambda_smooth",
    "lambda_expert",
    "ig_steps",
    "time_binning",
    "risk_score",
    "weight_decay",
    "architecture",
    "norm",
    "es_metric",
    "likelihood",
    "lambda_cox",
    "calibrate",
    "ema_decay",
    "input_noise",
    "input_gate",
)


class TEXGISAModel(BaseSurvivalModel):
    """
    TEXGISA: Time-dependent EXtreme Gradient Integration for Survival Analysis.

    A discrete-time survival network with time-dependent TEXGI attributions:

    1. Time is split into intervals (edges at quantiles of the event times); the
       network predicts one hazard per interval.
    2. A Cox risk head on the shared representation is trained with the Cox
       partial likelihood and shifts every interval's hazard logit
       (proportional-hazards coupling); its output is the risk score.
    3. The hazards are trained with the piecewise-constant-hazard likelihood,
       which uses the position of each time inside its interval.
    4. A temperature and a bias of the hazard logits are fitted on the validation
       data after training, so the survival curves are calibrated.
    5. Expert rules and temporal smoothness act as attribution priors.

    The defaults are the values most often selected by nested cross-validation
    in the benchmark of the accompanying software paper (see ``benchmarks/``).

    Parameters
    ----------
    hidden_layers : list of int, default [256, 128]
        'mlp': the widths of the hidden layers. 'resnet': the width is
        hidden_layers[0] and the number of residual blocks is len(hidden_layers).
    num_time_bins : int, default 20
        Number of time intervals (quantile binning may merge tied edges).
    dropout : float, default 0.2
        Dropout probability.
    activation : str, default 'selu'
        Activation function ('relu', 'elu', 'selu', 'gelu', 'tanh', 'sigmoid',
        'leaky_relu').
    norm : {'layer', 'batch'}, default 'layer'
        Normalisation layer of the backbone.
    architecture : {'resnet', 'mlp'}, default 'resnet'
        Residual MLP (Gorishniy et al., 2021) or a plain MLP.
    lambda_cox : float, default 1.0
        Weight of the Cox partial likelihood. 0 removes the Cox head; the risk
        score is then computed from the hazards (see `risk_score`).
    likelihood : {'pch', 'bce'}, default 'pch'
        Piecewise-constant-hazard likelihood, or the discrete-time likelihood
        (masked binary cross-entropy over the intervals at risk).
    calibrate : bool, default True
        Fit a temperature and a bias of the hazard logits on the validation data.
    ema_decay : float, default 0.9
        Decay of an exponential moving average of the weights; the averaged
        weights are validated and kept. 0 disables it.
    input_noise : float, default 0.1
        Standard deviation of Gaussian noise added to the standardised inputs
        during training.
    input_gate : bool, default False
        Re-weight the inputs with a learned feature gate.
    es_metric : {'loss', 'cindex'}, default 'loss'
        Validation quantity for early stopping and checkpointing.
    time_binning : {'quantile', 'uniform'}, default 'quantile'
        Interval edges at quantiles of the event times, or equal widths.
    risk_score : {'expected_time', 'hazard_sum'}, default 'expected_time'
        Risk score when there is no Cox head: minus the restricted mean survival
        time, or the sum of the interval hazards.
    weight_decay : float, default 0.01
        Decoupled weight decay (AdamW).
    lambda_smooth : float, default 0.0
        Weight of the temporal smoothness prior.
    lambda_expert : float, default 0.0
        Weight of the expert-rule prior.
    ig_steps : int, default 20
        Number of integration steps for TEXGI.
    device : str, default 'cpu'
        Device for computation.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        hidden_layers: List[int] = None,
        num_time_bins: int = 20,
        dropout: float = 0.2,
        activation: str = "selu",
        lambda_smooth: float = 0.0,
        lambda_expert: float = 0.0,
        ig_steps: int = 20,
        time_binning: str = "quantile",
        risk_score: str = "expected_time",
        weight_decay: float = 0.01,
        architecture: str = "resnet",
        norm: str = "layer",
        es_metric: str = "loss",
        likelihood: str = "pch",
        lambda_cox: float = 1.0,
        calibrate: bool = True,
        ema_decay: float = 0.9,
        input_noise: float = 0.1,
        input_gate: bool = False,
        device: str = "cpu",
        random_state: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(device=device, random_state=random_state)
        if kwargs:
            raise TypeError(f"Unknown parameters: {sorted(kwargs)}")
        if architecture not in ("mlp", "resnet"):
            raise ValueError("architecture must be 'mlp' or 'resnet'")
        if norm not in ("batch", "layer"):
            raise ValueError("norm must be 'batch' or 'layer'")
        if es_metric not in ("loss", "cindex"):
            raise ValueError("es_metric must be 'loss' or 'cindex'")
        if likelihood not in ("bce", "pch"):
            raise ValueError("likelihood must be 'bce' or 'pch'")
        if time_binning not in ("uniform", "quantile"):
            raise ValueError("time_binning must be 'uniform' or 'quantile'")
        if risk_score not in ("hazard_sum", "expected_time"):
            raise ValueError("risk_score must be 'hazard_sum' or 'expected_time'")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if lambda_cox < 0:
            raise ValueError("lambda_cox must be non-negative")

        self.hidden_layers = list(hidden_layers) if hidden_layers else [256, 128]
        self.num_time_bins = num_time_bins
        self.requested_time_bins = num_time_bins  # quantile binning may merge tied edges
        self.dropout = dropout
        self.activation = activation
        self.lambda_smooth = lambda_smooth
        self.lambda_expert = lambda_expert
        self.ig_steps = ig_steps
        self.time_binning = time_binning
        self.risk_score = risk_score
        self.weight_decay = weight_decay
        self.architecture = architecture
        self.norm = norm
        self.es_metric = es_metric
        self.likelihood = likelihood
        self.lambda_cox = float(lambda_cox)
        self.calibrate = bool(calibrate)
        self.ema_decay = float(ema_decay)
        self.input_noise = float(input_noise)
        self.input_gate = bool(input_gate)

        self.scaler = StandardScaler()
        self.time_bins = None
        self.generator = None
        self.expert_rules = []
        self.feature_names = None

    def _build_network(self, input_dim: int) -> nn.Module:
        """Backbone, optional input gate and optional Cox head."""
        gate = _FeatureGate(input_dim) if self.input_gate else None
        if self.architecture == "resnet":
            base = ResNetSurvivalNet(
                input_dim,
                self.num_time_bins,
                width=self.hidden_layers[0],
                n_blocks=max(1, len(self.hidden_layers)),
                dropout=self.dropout,
                norm=self.norm,
                activation=self.activation,
            )
        else:
            base = MultiTaskSurvivalNet(
                input_dim=input_dim,
                hidden_layers=self.hidden_layers,
                num_time_bins=self.num_time_bins,
                dropout=self.dropout,
                activation=self.activation,
                norm=self.norm,
            )
        return SurvivalModule(base, cox_head=self.lambda_cox > 0, input_gate=gate).to(self.device)

    def _make_time_bins(self, time: np.ndarray, event: np.ndarray) -> np.ndarray:
        """Interval edges covering [min(time), max(time)]."""
        k = self.requested_time_bins
        lo, hi = float(time.min()), float(time.max())
        if self.time_binning == "quantile" and (event == 1).sum() >= k:
            inner = np.quantile(time[event == 1], np.linspace(0, 1, k + 1)[1:-1])
            edges = np.unique(np.concatenate([[lo], inner[(inner > lo) & (inner < hi)], [hi]]))
            if len(edges) >= 3:
                return edges
        return np.linspace(lo, hi, k + 1)

    def add_expert_rule(
        self, feature: str, relation: str, threshold: str, sign: int = 1, weight: float = 1.0
    ):
        """Add expert knowledge rule for feature importance constraints."""
        rule = {
            "feature": feature,
            "relation": relation,
            "threshold": threshold,
            "sign": sign,
            "weight": weight,
        }
        self.expert_rules.append(rule)

    def fit(
        self,
        X: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        validation_data: Optional[Tuple] = None,
        epochs: int = 300,
        batch_size: int = 512,
        learning_rate: float = 0.003,
        early_stopping: bool = True,
        patience: int = 15,
        verbose: int = 1,
        expert_rules: Optional[Dict] = None,
        validation_split: float = 0.2,
        **kwargs,
    ) -> None:
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
            (X_val, time_val, event_val), used for early stopping, checkpointing
            and calibration
        epochs : int, default 300
            Maximum number of training epochs
        batch_size : int, default 512
            Batch size
        learning_rate : float, default 0.003
            Learning rate
        early_stopping : bool
            Whether to use early stopping
        patience : int, default 15
            Early stopping patience (epochs)
        verbose : int
            Verbosity level
        expert_rules : dict, optional
            Expert rules dictionary
        validation_split : float, default 0.2
            If `validation_data` is None, this fraction of the training data
            (chosen at random with `random_state`) is held out as validation
            data. 0 trains on all data for `epochs` epochs without validation.
        **kwargs
            Additional parameters
        """
        # Store feature names if available (the TexGISa wrapper passes them
        # explicitly because it hands over a plain array)
        feature_names = kwargs.pop("feature_names", None)
        if hasattr(X, "columns"):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        time = np.asarray(time, dtype=float)
        event = np.asarray(event)
        X = np.asarray(X)
        if validation_data is None and validation_split > 0:
            n_val = int(round(len(time) * validation_split))
            if not 0 < n_val < len(time):
                raise ValueError("validation_split leaves no training or validation data")
            idx = np.random.default_rng(self.random_state).permutation(len(time))
            va, tr = idx[:n_val], idx[n_val:]
            validation_data = (X[va], time[va], event[va])
            X, time, event = X[tr], time[tr], event[tr]

        # Normalize features
        X = self.scaler.fit_transform(X.astype(np.float32))

        # Create time discretization
        self.time_bins = self._make_time_bins(time, event)
        self.num_time_bins = len(self.time_bins) - 1

        # Discretize survival times
        time_discrete = np.digitize(time, self.time_bins[1:])
        time_discrete = np.clip(time_discrete, 0, self.num_time_bins - 1)

        # Create supervision labels and masks
        labels, masks = self._create_supervision(time_discrete, event)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        labels_tensor = torch.FloatTensor(labels).to(self.device)
        masks_tensor = torch.FloatTensor(masks).to(self.device)
        bin_tensor = torch.as_tensor(time_discrete, dtype=torch.long, device=self.device)
        event_tensor = torch.as_tensor(event, dtype=torch.float32, device=self.device)
        time_tensor = torch.as_tensor(time, dtype=torch.float32, device=self.device)
        frac_tensor = torch.as_tensor(
            self._fraction_in_bin(time, time_discrete), dtype=torch.float32, device=self.device
        )

        # Create model
        input_dim = X.shape[1]
        self.model = self._build_network(input_dim)

        # The estimator explains against the per-feature upper-quantile baseline;
        # the adversarial extreme-baseline generator is used by MySATrainer.

        # Set up expert rules
        if expert_rules and "rules" in expert_rules:
            self.expert_rules = expert_rules["rules"]
        # Fail fast on rules that name unknown features; the per-batch loss
        # skips failing rules, which used to hide such typos completely.
        for rule in self.expert_rules:
            self._get_feature_index(rule["feature"])

        # Training setup
        if self.weight_decay > 0:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience // 2, factor=0.5
        )

        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        # Exponential moving average of the weights (validated and kept instead of the raw ones)
        ema_state = (
            {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            if self.ema_decay > 0
            else None
        )

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            # Mini-batch training
            n_samples = X_tensor.shape[0]
            indices = torch.randperm(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i : i + batch_size]
                if len(batch_indices) < 2:
                    continue  # BatchNorm cannot train on a single sample
                X_batch = X_tensor[batch_indices]
                if self.input_noise > 0:
                    X_batch = X_batch + self.input_noise * torch.randn_like(X_batch)
                labels_batch = labels_tensor[batch_indices]
                masks_batch = masks_tensor[batch_indices]

                optimizer.zero_grad()

                # Forward pass
                hazards, cox_risk = self.model.hazards_and_risk(X_batch)

                # Main loss: discrete-time likelihood (masked BCE) or the
                # piecewise-constant-hazard likelihood that uses within-bin times
                if self.likelihood == "pch":
                    main_loss = self._pch_loss(
                        hazards,
                        bin_tensor[batch_indices],
                        event_tensor[batch_indices],
                        frac_tensor[batch_indices],
                    )
                else:
                    main_loss = self._masked_bce_loss(hazards, labels_batch, masks_batch)
                if cox_risk is not None:
                    main_loss = main_loss + self.lambda_cox * self._cox_loss(
                        cox_risk, time_tensor[batch_indices], event_tensor[batch_indices]
                    )

                # Regularization losses
                smooth_loss = self._temporal_smoothness_loss(hazards)
                expert_loss = torch.tensor(0.0, device=self.device)

                if self.lambda_expert > 0 and self.expert_rules:
                    expert_loss = self._compute_expert_loss(X_batch, hazards)

                total_loss = (
                    main_loss + self.lambda_smooth * smooth_loss + self.lambda_expert * expert_loss
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        for k, v in self.model.state_dict().items():
                            if v.dtype.is_floating_point:
                                ema_state[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
                            else:
                                ema_state[k].copy_(v)

                epoch_loss += total_loss.item()

            epoch_loss /= n_samples // batch_size + 1

            # Validation
            if validation_data is not None:
                train_state = None
                if ema_state is not None:  # validate (and checkpoint) the averaged weights
                    train_state = {
                        k: v.detach().clone() for k, v in self.model.state_dict().items()
                    }
                    self.model.load_state_dict(ema_state)
                val_loss = self._validate(validation_data)
                scheduler.step(val_loss)
                if self.es_metric == "cindex":
                    val_loss = -self._validation_cindex(validation_data)  # lower is better

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")
                if train_state is not None:
                    self.model.load_state_dict(train_state)  # continue training the raw weights

                if early_stopping and patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss={epoch_loss:.4f}")

        # Keep the weights of the best validation epoch, not the last one
        if best_state is not None:
            self.model.load_state_dict(best_state)
        elif ema_state is not None:
            self.model.load_state_dict(ema_state)
        if self.calibrate and validation_data is not None:
            self._fit_calibration(validation_data)

        self.is_fitted = True

        if verbose:
            print("Initializing TEXGI explainer...")
        # Save a sample of training data for importance computation
        self._init_explainer(X_tensor[: min(200, len(X_tensor))].detach())
        if verbose:
            print("TEXGI explainer initialized")

    def _init_explainer(self, reference: torch.Tensor) -> None:
        """Build the attribution explainer from a (standardized) reference sample."""
        from ..explainers.texgi import TEXGIExplainer

        self.training_data_sample = reference.to(self.device)
        background_size = min(100, len(reference))
        background_indices = torch.randperm(len(reference))[:background_size]
        self.texgi_explainer = TEXGIExplainer(
            background_data=self.training_data_sample[background_indices],
            batch_size=min(16, background_size),
            random_alpha=True,
            k=max(1, self.ig_steps // 5),  # Use ig_steps/5 reference samples
            scale_by_inputs=True,
            device=self.device,
        )

    def predict_survival(self, X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict survival probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Preprocess
        if hasattr(X, "values"):
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

    def predict_hazard(self, X: np.ndarray, times: Optional[np.ndarray] = None) -> np.ndarray:
        """Discrete-time hazards per model bin, or the generic approximation at `times`."""
        if times is None:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            return self._predict_hazards(X)
        return super().predict_hazard(X, times)

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        if getattr(self.model, "cox", None) is not None:
            return self._cox_risk(X)
        return self._risk_from_hazards(self._predict_hazards(X))

    def _cox_risk(self, X) -> np.ndarray:
        """Output of the Cox risk head (higher = earlier event)."""
        if hasattr(X, "values"):
            X = X.values
        Xt = torch.FloatTensor(self.scaler.transform(np.asarray(X, dtype=np.float32)))
        self.model.eval()
        with torch.no_grad():
            return self.model.risk(Xt.to(self.device)).cpu().numpy()

    def _risk_from_hazards(self, hazards: np.ndarray) -> np.ndarray:
        if getattr(self, "risk_score", "hazard_sum") == "expected_time":
            # minus the restricted mean survival time: area under S(t) on the
            # model's grid, with S = 1 at the first edge and linear in between
            S = np.hstack([np.ones((len(hazards), 1)), np.cumprod(1 - hazards, axis=1)])
            widths = np.diff(np.asarray(self.time_bins, dtype=float))
            return -(0.5 * (S[:, :-1] + S[:, 1:]) * widths).sum(axis=1)
        return hazards.sum(axis=1)

    def _validation_cindex(self, validation_data) -> float:
        """Harrell's C-index of the current network on the validation data."""
        from ..metrics import concordance_index

        X_val, time_val, event_val = validation_data
        if hasattr(X_val, "values"):
            X_val = X_val.values
        if getattr(self.model, "cox", None) is not None:
            risk = self._cox_risk(X_val)
        else:
            X_val = torch.FloatTensor(self.scaler.transform(np.asarray(X_val, dtype=np.float32)))
            self.model.eval()
            with torch.no_grad():
                hazards = self.model(X_val.to(self.device)).cpu().numpy()
            risk = self._risk_from_hazards(hazards)
        if not np.all(np.isfinite(risk)):
            return 0.0
        return float(concordance_index(np.asarray(time_val), risk, np.asarray(event_val)))

    def get_feature_importance(
        self, method: str = "texgi", n_steps: int = None, **kwargs
    ) -> Dict[str, np.ndarray]:
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
        if method == "texgi" or method == "default":
            return self._compute_texgi_importance(n_steps or self.ig_steps)
        elif method == "weights":
            return self._get_weight_importance()
        elif method == "permutation":
            return self._get_permutation_importance(**kwargs)
        else:
            raise ValueError(f"Unknown importance method: {method}")

    def _create_supervision(self, time_discrete, event):
        """Create supervision labels and masks for discrete time bins."""
        n_samples = len(time_discrete)
        labels = np.zeros((n_samples, self.num_time_bins))
        masks = np.zeros((n_samples, self.num_time_bins))

        # Convert to numpy array if it's a pandas Series
        if hasattr(event, "values"):
            event = event.values

        for i in range(n_samples):
            t_idx = time_discrete[i]
            # Mask all time bins up to and including the event/censoring time
            masks[i, : t_idx + 1] = 1
            # If event occurred, set hazard to 1 at that time
            if event[i] == 1:
                labels[i, t_idx] = 1

        return labels, masks

    def _masked_bce_loss(self, hazards, labels, masks):
        """Compute masked binary cross entropy loss."""
        eps = 1e-7
        hazards = torch.clamp(hazards, eps, 1 - eps)
        bce = -(labels * torch.log(hazards) + (1 - labels) * torch.log(1 - hazards))
        masked_loss = (bce * masks).sum() / masks.sum().clamp_min(1.0)
        return masked_loss

    def _fraction_in_bin(self, time: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Position of each time inside its interval, in [0, 1]."""
        lo = self.time_bins[bins]
        width = np.maximum(self.time_bins[bins + 1] - lo, 1e-12)
        return np.clip((np.asarray(time, dtype=float) - lo) / width, 0.0, 1.0)

    def _pch_loss(self, hazards, bins, events, frac):
        """
        Negative log-likelihood of a piecewise-constant hazard model (Kvamme &
        Borgan, 2021). With interval cumulative hazards L_j = -log(1 - h_j), a
        subject in interval k at relative position u contributes
        -[e * log L_k - u * L_k - sum_{j<k} L_j] (constant terms dropped).
        """
        eps = 1e-7
        L = -torch.log(torch.clamp(1 - hazards, eps, 1.0))
        cum_before = torch.cumsum(L, dim=1) - L  # sum_{j<k} L_j
        Lk = L.gather(1, bins.unsqueeze(1)).squeeze(1)
        before = cum_before.gather(1, bins.unsqueeze(1)).squeeze(1)
        ll = events * torch.log(Lk + eps) - frac * Lk - before
        return -ll.mean()

    @staticmethod
    def _cox_loss(risk, time, event):
        """Negative Cox partial log-likelihood (Breslow ties) on a mini-batch."""
        if event.sum() < 1:
            return risk.new_zeros(())
        order = torch.argsort(time, descending=True)
        r, e = risk[order], event[order]
        log_risk_set = torch.logcumsumexp(r, dim=0)  # risk set: all with time >= t_i
        return -((r - log_risk_set) * e).sum() / e.sum()

    def _fit_calibration(self, validation_data, steps: int = 100) -> None:
        """Fit a temperature and bias on the hazard logits (masked BCE on validation)."""
        X_val, time_val, event_val = validation_data
        if hasattr(X_val, "values"):
            X_val = X_val.values
        Xv = torch.FloatTensor(self.scaler.transform(np.asarray(X_val, dtype=np.float32)))
        t_disc = np.clip(
            np.digitize(np.asarray(time_val), self.time_bins[1:]), 0, self.num_time_bins - 1
        )
        labels, masks = self._create_supervision(t_disc, np.asarray(event_val))
        labels = torch.FloatTensor(labels).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model.raw_logits(Xv.to(self.device))
        log_t = torch.zeros((), requires_grad=True, device=self.device)
        bias = torch.zeros((), requires_grad=True, device=self.device)
        opt = torch.optim.LBFGS([log_t, bias], lr=0.5, max_iter=steps)

        def closure():
            opt.zero_grad()
            h = _softplus_hazard(logits / torch.exp(log_t) + bias)
            loss = self._masked_bce_loss(h, labels, masks)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            self.model.temperature.copy_(torch.exp(log_t).clamp(0.05, 20.0))
            self.model.logit_bias.copy_(bias.clamp(-5.0, 5.0))

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

        # Time-dependent attributions of the current batch with respect to an
        # upper-tail (extreme) baseline. create_graph=True keeps the attributions
        # differentiable in the model weights, so the rule penalty below actually
        # shapes training (previously it was computed after fit() under no_grad
        # and never reached the optimizer).
        from ..texgi_utils import texgi_time_series

        if X.shape[0] < 2:
            return torch.tensor(0.0, device=X.device)
        # Eval mode keeps BatchNorm/Dropout deterministic along the integration
        # path; gradients still flow to the weights.
        was_training = self.model.training
        self.model.eval()
        try:
            phi = texgi_time_series(
                self.model,
                X.detach(),
                X_baseline=None,  # per-feature 98th-percentile baseline
                M=min(self.ig_steps, 20),
                t_sample=min(10, self.num_time_bins),
                create_graph=True,
            )  # [T', B, D]
        finally:
            self.model.train(was_training)
        attributions = phi.mean(dim=0)  # [batch_size, n_features]

        expert_loss = torch.tensor(0.0, device=X.device)

        for rule in self.expert_rules:
            try:
                # Get feature index
                feature_idx = self._get_feature_index(rule["feature"])

                feature_attr = attributions[:, feature_idx]  # [batch_size]
                feature_values = X[:, feature_idx]

                # Compute threshold
                threshold = self._compute_threshold(rule["threshold"], X, feature_idx)

                # Create mask for samples satisfying the condition
                relation = rule["relation"]
                if relation == ">":
                    mask = feature_values > threshold
                elif relation == ">=":
                    mask = feature_values >= threshold
                elif relation == "<":
                    mask = feature_values < threshold
                elif relation == "<=":
                    mask = feature_values <= threshold
                elif relation == "==":
                    mask = torch.abs(feature_values - threshold) < 1e-4
                else:
                    continue

                # For samples satisfying condition, check if attribution matches expected sign
                if mask.any():
                    expected_sign = rule["sign"]  # +1 or -1
                    actual_attr = feature_attr[mask]

                    # Penalize when attribution sign doesn't match expectation
                    # Loss is high when attribution has opposite sign
                    sign_penalty = torch.where(
                        actual_attr * expected_sign < 0,  # Wrong sign
                        torch.abs(actual_attr),
                        torch.zeros_like(actual_attr),
                    )

                    # Weight by rule importance
                    rule_loss = (sign_penalty * rule["weight"]).mean()
                    expert_loss += rule_loss

            except Exception as e:
                # Skip problematic rules
                continue

        return expert_loss

    def _train_generator(self, X_tensor, epochs=100, verbose=False):
        """Train adversarial generator for extreme baselines."""
        if self.generator is None:
            self.generator = AdversarialGenerator(input_dim=X_tensor.shape[1]).to(self.device)

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
                hazards = (
                    self.model(X_adv)
                    if self.model
                    else torch.zeros(batch_size, self.num_time_bins, device=self.device)
                )
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

        if not hasattr(self, "texgi_explainer"):
            raise ValueError(
                "TEXGI explainer not initialized. Model may not have been trained properly."
            )

        # Use stored training data sample, but match the explainer's batch size
        explainer_batch_size = self.texgi_explainer.batch_size
        X_sample = self.training_data_sample[:explainer_batch_size].to(self.device)

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
                return [output[:, i : i + 1] for i in range(self.num_time_bins)]

        wrapped_model = ModelWrapper(self.model, self.num_time_bins)
        wrapped_model.eval()

        # Compute TEXGI attributions in batches to get more stable estimates
        all_attributions = []
        num_batches = min(5, len(self.training_data_sample) // explainer_batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * explainer_batch_size
            end_idx = start_idx + explainer_batch_size
            X_batch = self.training_data_sample[start_idx:end_idx].to(self.device)

            # Note: TEXGI needs gradients, so we don't use torch.no_grad() here
            attributions_list = self.texgi_explainer.compute_attributions(
                model=wrapped_model, input_tensor=X_batch, sparse_labels=None
            )

            # Average across time bins for this batch
            avg_attributions = sum(attributions_list) / len(attributions_list)
            all_attributions.append(avg_attributions.detach())

        # Concatenate all batches
        all_attributions = torch.cat(all_attributions, dim=0)

        # Convert to numpy and compute statistics
        avg_attributions_np = all_attributions.cpu().numpy()

        # Compute mean and std across samples
        importance = np.abs(avg_attributions_np).mean(axis=0)
        std = np.abs(avg_attributions_np).std(axis=0)

        # Normalize to sum to 1
        importance = importance / (importance.sum() + 1e-10)

        return {"importance": importance, "std": std}

    def _hazards_to_survival(self, hazards):
        """Convert discrete hazards to survival probabilities."""
        # S(t) = ∏(1 - h(s)) for s ≤ t
        survival = np.cumprod(1 - hazards, axis=1)
        return survival

    def _predict_hazards(self, X):
        """Predict hazards for input data."""
        if hasattr(X, "values"):
            X = X.values
        X = self.scaler.transform(X.astype(np.float32))
        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            hazards = self.model(X_tensor).cpu().numpy()

        return hazards

    def _interpolate_survival(self, survival_probs, times):
        """Interpolate survival probabilities to requested times."""
        # Column k of survival_probs is S at the right edge of bin k, and
        # S = 1 at the left edge of the first bin. Interpolate linearly between
        # edges; hold the end values outside the grid.
        edges = np.asarray(self.time_bins, dtype=float)
        n = survival_probs.shape[0]
        S_nodes = np.hstack([np.ones((n, 1)), survival_probs])  # [n, T+1] at edges
        times = np.asarray(times, dtype=float)
        idx = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)
        t0, t1 = edges[idx], edges[idx + 1]
        w = np.clip((times - t0) / np.maximum(t1 - t0, 1e-12), 0.0, 1.0)
        return (1 - w) * S_nodes[:, idx] + w * S_nodes[:, idx + 1]

    def _validate(self, validation_data):
        """Compute validation loss."""
        X_val, time_val, event_val = validation_data

        if hasattr(X_val, "values"):
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
            hazards, cox_risk = self.model.hazards_and_risk(X_tensor)
            loss = self._masked_bce_loss(hazards, labels_tensor, masks_tensor)
            if cox_risk is not None:
                loss = loss + self.lambda_cox * self._cox_loss(
                    cox_risk,
                    torch.as_tensor(np.asarray(time_val), dtype=torch.float32, device=self.device),
                    torch.as_tensor(np.asarray(event_val), dtype=torch.float32, device=self.device),
                )

        return loss.item()

    def _get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for saving."""
        import pickle

        # Serialize scaler
        scaler_bytes = None
        if hasattr(self.scaler, "mean_"):  # Check if scaler is fitted
            scaler_bytes = pickle.dumps(self.scaler)

        return {
            **{key: getattr(self, key) for key in _HYPERPARAMETERS},
            "expert_rules": self.expert_rules,
            "feature_names": self.feature_names,
            "time_bins": self.time_bins.tolist() if self.time_bins is not None else None,
            "scaler": scaler_bytes,
            "training_data_sample": (
                self.training_data_sample.cpu().numpy()
                if getattr(self, "training_data_sample", None) is not None
                else None
            ),
        }

    def _create_model_from_config(self, config: Dict[str, Any]) -> None:
        """Create model from configuration."""
        import pickle

        # Restore hyperparameters first: attribution and prediction code read them
        for key in _HYPERPARAMETERS:
            if key in config:
                setattr(self, key, config[key])

        input_dim = len(config.get("feature_names", []))
        if input_dim > 0:
            self.model = self._build_network(input_dim)

            self.generator = AdversarialGenerator(input_dim).to(self.device)

            # Restore other attributes
            self.expert_rules = config.get("expert_rules", [])
            self.feature_names = config.get("feature_names", [])
            if "time_bins" in config and config["time_bins"]:
                self.time_bins = np.array(config["time_bins"])

            # Restore scaler
            if "scaler" in config and config["scaler"] is not None:
                self.scaler = pickle.loads(config["scaler"])
            else:
                self.scaler = StandardScaler()

            if config.get("training_data_sample") is not None:
                self._init_explainer(
                    torch.as_tensor(config["training_data_sample"], dtype=torch.float32)
                )

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
        """Compute threshold value from specification.

        X is the standardized batch, so a numeric threshold given in the
        feature's original units is mapped onto the standardized scale.
        """
        if isinstance(threshold, (int, float, np.integer, np.floating)):
            if hasattr(self.scaler, "mean_"):
                return float(
                    (threshold - self.scaler.mean_[feature_idx]) / self.scaler.scale_[feature_idx]
                )
            return float(threshold)
        elif threshold == "mean":
            return X[:, feature_idx].mean().item()
        elif threshold == "median":
            return X[:, feature_idx].median().item()
        elif threshold == "min":
            return X[:, feature_idx].min().item()
        elif threshold == "max":
            return X[:, feature_idx].max().item()
        else:
            raise ValueError(f"Unknown threshold specification: {threshold}")
