# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Baseline survival models (CoxTime, DeepSurv, DeepHit) backed by pycox.

These classes extend :class:`BaseSurvivalModel` and lazily import
``pycox`` / ``torchtuples`` so the rest of the library works without them.
Install the optional *baselines* extra to use these models::

    pip install texgisa-survival[baselines]
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base import BaseSurvivalModel


def _require_pycox():
    """Lazy-import pycox and torchtuples with a helpful error message."""
    try:
        import pycox  # noqa: F401
        import torchtuples  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pycox and torchtuples are required for baseline models. "
            "Install them with:  pip install texgisa-survival[baselines]"
        ) from exc

    # pycox uses scipy.integrate.simps which was removed in scipy >=1.14.
    # Patch it back so pycox's EvalSurv methods keep working.
    import scipy.integrate
    if not hasattr(scipy.integrate, "simps"):
        scipy.integrate.simps = scipy.integrate.simpson


def _encode_non_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode any non-numeric columns."""
    non_numeric = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if non_numeric:
        X = pd.get_dummies(X, columns=non_numeric, drop_first=False)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _prepare_arrays(df: pd.DataFrame, feature_cols: List[str]):
    """Return (x_float32, (durations_float32, events_float32)) from a DataFrame."""
    x = df[feature_cols].astype("float32").values
    durations = df["duration"].astype("float32").values
    events = df["event"].astype("float32").values
    return x, (durations, events)


class CoxTimeModel(BaseSurvivalModel):
    """CoxTime model backed by pycox.

    Parameters
    ----------
    num_nodes : list of int
        Hidden layer sizes for the MLP.
    batch_norm : bool
        Whether to use batch normalisation.
    dropout : float
        Dropout probability.
    device : str
        ``'cpu'`` or ``'cuda'``.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        num_nodes: Optional[List[int]] = None,
        batch_norm: bool = True,
        dropout: float = 0.1,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ):
        super().__init__(device=device, random_state=random_state)
        self.num_nodes = num_nodes or [32, 32]
        self.batch_norm = batch_norm
        self.dropout = dropout
        self._pycox_model = None
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # BaseSurvivalModel interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X,
        time,
        event,
        validation_data=None,
        epochs: int = 512,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        _require_pycox()
        os.environ["PYCOX_NO_DATA"] = "1"
        import torchtuples as tt
        from pycox.models import CoxTime
        from pycox.models.cox_time import MLPVanillaCoxTime

        X, time, event = self._to_dataframe(X, time, event)
        X = _encode_non_numeric(X)
        self._feature_cols = list(X.columns)

        if validation_data is not None:
            Xv, tv, ev = validation_data
            Xv, tv, ev = self._to_dataframe(Xv, tv, ev)
            Xv = _encode_non_numeric(Xv)
            # Align columns
            for c in self._feature_cols:
                if c not in Xv.columns:
                    Xv[c] = 0.0
            Xv = Xv[self._feature_cols]
            x_train = X.astype("float32").values
            y_train = (time.astype("float32").values, event.astype("float32").values)
            x_val = Xv.astype("float32").values
            y_val = (tv.astype("float32").values, ev.astype("float32").values)
        else:
            # Auto train/val split
            df = pd.concat([X, time.rename("duration"), event.rename("event")], axis=1).reset_index(drop=True)
            train_idx, val_idx = train_test_split(
                df.index, test_size=0.2, stratify=df["event"], random_state=42,
            )
            train_df = df.loc[train_idx]
            val_df = df.loc[val_idx]
            x_train, y_train = _prepare_arrays(train_df, self._feature_cols)
            x_val, y_val = _prepare_arrays(val_df, self._feature_cols)

        in_features = x_train.shape[1]
        net = MLPVanillaCoxTime(
            in_features=in_features,
            num_nodes=self.num_nodes,
            batch_norm=self.batch_norm,
            dropout=self.dropout,
        )
        model = CoxTime(net, tt.optim.Adam)
        model.optimizer.set_lr(learning_rate)

        callbacks = [tt.callbacks.EarlyStopping(patience=patience)] if early_stopping else []
        model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            val_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=bool(verbose),
        )
        model.compute_baseline_hazards()
        self._pycox_model = model
        self.is_fitted = True

    def predict_survival(self, X, times=None) -> np.ndarray:
        self._check_fitted()
        x = self._prepare_input(X)
        surv_df = self._pycox_model.predict_surv_df(x)
        surv = surv_df.values.T  # [n_samples, n_times]
        if times is not None:
            surv = self._interpolate_surv(surv, surv_df.index.values, np.asarray(times))
        return surv

    def predict_risk(self, X) -> np.ndarray:
        surv = self.predict_survival(X)
        return 1.0 - surv.mean(axis=1)

    def evaluate(self, X, time, event) -> Dict[str, float]:
        """Compute C-index, IBS, and NLL on the given data."""
        _require_pycox()
        from pycox.evaluation import EvalSurv
        x = self._prepare_input(X)
        surv_df = self._pycox_model.predict_surv_df(x)
        durations = np.asarray(time, dtype="float32")
        events = np.asarray(event, dtype="float32")
        ev = EvalSurv(surv_df, durations, events, censor_surv="km")
        time_grid = np.linspace(max(1, durations.min()), durations.max(), 100)
        return {
            "c_index": ev.concordance_td(),
            "integrated_brier_score": ev.integrated_brier_score(time_grid),
            "integrated_nbll": ev.integrated_nbll(time_grid),
        }

    # ------------------------------------------------------------------
    # Save / Load helpers
    # ------------------------------------------------------------------

    def _get_model_config(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "feature_cols": self._feature_cols,
        }

    def _create_model_from_config(self, config: Dict[str, Any]) -> None:
        self.num_nodes = config.get("num_nodes", [32, 32])
        self.batch_norm = config.get("batch_norm", True)
        self.dropout = config.get("dropout", 0.1)
        self._feature_cols = config.get("feature_cols", [])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self.is_fitted or self._pycox_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    @staticmethod
    def _to_dataframe(X, time, event):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        else:
            X = pd.DataFrame(X).reset_index(drop=True)
        time = pd.Series(np.asarray(time, dtype=float), name="duration")
        event = pd.Series(np.asarray(event, dtype=float), name="event")
        return X.reset_index(drop=True), time.reset_index(drop=True), event.reset_index(drop=True)

    def _prepare_input(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        X = _encode_non_numeric(pd.DataFrame(X).reset_index(drop=True))
        for c in self._feature_cols:
            if c not in X.columns:
                X[c] = 0.0
        return X[self._feature_cols].astype("float32").values

    @staticmethod
    def _interpolate_surv(surv: np.ndarray, index_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
        """Linearly interpolate survival matrix at ``query_times``."""
        out = np.zeros((surv.shape[0], len(query_times)))
        for i, t in enumerate(query_times):
            idx = np.searchsorted(index_times, t, side="right") - 1
            idx = max(0, min(idx, surv.shape[1] - 1))
            out[:, i] = surv[:, idx]
        return out


class DeepSurvModel(BaseSurvivalModel):
    """DeepSurv (CoxPH neural network) model backed by pycox.

    Parameters
    ----------
    num_nodes : list of int
        Hidden layer sizes for the MLP.
    batch_norm : bool
        Whether to use batch normalisation.
    dropout : float
        Dropout probability.
    device : str
        ``'cpu'`` or ``'cuda'``.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        num_nodes: Optional[List[int]] = None,
        batch_norm: bool = True,
        dropout: float = 0.1,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ):
        super().__init__(device=device, random_state=random_state)
        self.num_nodes = num_nodes or [32, 32]
        self.batch_norm = batch_norm
        self.dropout = dropout
        self._pycox_model = None
        self._feature_cols: List[str] = []

    def fit(
        self,
        X,
        time,
        event,
        validation_data=None,
        epochs: int = 64,
        batch_size: int = 256,
        learning_rate: float = 0.1,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        _require_pycox()
        os.environ["PYCOX_NO_DATA"] = "1"
        import torchtuples as tt
        from pycox.models import CoxPH

        X, time, event = CoxTimeModel._to_dataframe(X, time, event)
        X = _encode_non_numeric(X)
        self._feature_cols = list(X.columns)

        if validation_data is not None:
            Xv, tv, ev = validation_data
            Xv, tv, ev = CoxTimeModel._to_dataframe(Xv, tv, ev)
            Xv = _encode_non_numeric(Xv)
            for c in self._feature_cols:
                if c not in Xv.columns:
                    Xv[c] = 0.0
            Xv = Xv[self._feature_cols]
            x_train = X.astype("float32").values
            y_train = (time.astype("float32").values, event.astype("float32").values)
            x_val = Xv.astype("float32").values
            y_val = (tv.astype("float32").values, ev.astype("float32").values)
        else:
            df = pd.concat([X, time.rename("duration"), event.rename("event")], axis=1).reset_index(drop=True)
            train_idx, val_idx = train_test_split(
                df.index, test_size=0.2, stratify=df["event"], random_state=42,
            )
            x_train, y_train = _prepare_arrays(df.loc[train_idx], self._feature_cols)
            x_val, y_val = _prepare_arrays(df.loc[val_idx], self._feature_cols)

        in_features = x_train.shape[1]
        net = tt.practical.MLPVanilla(
            in_features, self.num_nodes, out_features=1,
            batch_norm=self.batch_norm, dropout=self.dropout,
        )
        model = CoxPH(net, tt.optim.Adam)
        model.optimizer.set_lr(learning_rate)

        callbacks = [tt.callbacks.EarlyStopping(patience=patience)] if early_stopping else []
        model.fit(
            x_train, y_train,
            batch_size, epochs, callbacks, bool(verbose),
            val_data=(x_val, y_val), val_batch_size=batch_size,
        )
        model.compute_baseline_hazards()
        self._pycox_model = model
        self.is_fitted = True

    def predict_survival(self, X, times=None) -> np.ndarray:
        self._check_fitted()
        x = self._prepare_input(X)
        surv_df = self._pycox_model.predict_surv_df(x)
        surv = surv_df.values.T
        if times is not None:
            surv = CoxTimeModel._interpolate_surv(surv, surv_df.index.values, np.asarray(times))
        return surv

    def predict_risk(self, X) -> np.ndarray:
        surv = self.predict_survival(X)
        return 1.0 - surv.mean(axis=1)

    def evaluate(self, X, time, event) -> Dict[str, float]:
        _require_pycox()
        from pycox.evaluation import EvalSurv
        x = self._prepare_input(X)
        surv_df = self._pycox_model.predict_surv_df(x)
        durations = np.asarray(time, dtype="float32")
        events = np.asarray(event, dtype="float32")
        ev = EvalSurv(surv_df, durations, events, censor_surv="km")
        time_grid = np.linspace(max(1, durations.min()), durations.max(), 100)
        return {
            "c_index": ev.concordance_td(),
            "integrated_brier_score": ev.integrated_brier_score(time_grid),
            "integrated_nbll": ev.integrated_nbll(time_grid),
        }

    def _get_model_config(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "feature_cols": self._feature_cols,
        }

    def _create_model_from_config(self, config: Dict[str, Any]) -> None:
        self.num_nodes = config.get("num_nodes", [32, 32])
        self.batch_norm = config.get("batch_norm", True)
        self.dropout = config.get("dropout", 0.1)
        self._feature_cols = config.get("feature_cols", [])

    def _check_fitted(self):
        if not self.is_fitted or self._pycox_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def _prepare_input(self, X) -> np.ndarray:
        return CoxTimeModel._prepare_input(self, X)


class DeepHitModel(BaseSurvivalModel):
    """DeepHit (single-risk) model backed by pycox.

    Parameters
    ----------
    num_nodes : list of int
        Hidden layer sizes for the MLP.
    batch_norm : bool
        Whether to use batch normalisation.
    dropout : float
        Dropout probability.
    num_intervals : int
        Number of discrete time intervals for discretisation.
    out_features : int
        Output dimension of the network (PMF size).
    alpha : float
        Weight for the ranking loss component.
    sigma : float
        Bandwidth for the ranking loss kernel.
    device : str
        ``'cpu'`` or ``'cuda'``.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        num_nodes: Optional[List[int]] = None,
        batch_norm: bool = True,
        dropout: float = 0.01,
        num_intervals: int = 20,
        out_features: int = 4000,
        alpha: float = 0.2,
        sigma: float = 0.1,
        device: str = "cpu",
        random_state: Optional[int] = None,
    ):
        super().__init__(device=device, random_state=random_state)
        self.num_nodes = num_nodes or [32, 32]
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.num_intervals = num_intervals
        self.out_features = out_features
        self.alpha = alpha
        self.sigma = sigma
        self._pycox_model = None
        self._feature_cols: List[str] = []
        self._time_bins: Optional[np.ndarray] = None

    def fit(
        self,
        X,
        time,
        event,
        validation_data=None,
        epochs: int = 512,
        batch_size: int = 64,
        learning_rate: float = 0.1,
        early_stopping: bool = True,
        patience: int = 10,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        _require_pycox()
        os.environ["PYCOX_NO_DATA"] = "1"
        import torchtuples as tt
        from pycox.models import DeepHitSingle

        X, time_s, event_s = CoxTimeModel._to_dataframe(X, time, event)
        X = _encode_non_numeric(X)
        self._feature_cols = list(X.columns)

        # Build full df for splitting
        df = pd.concat([X, time_s.rename("duration"), event_s.rename("event")], axis=1).reset_index(drop=True)
        self._time_bins = np.linspace(df["duration"].min(), df["duration"].max(), self.num_intervals)

        def _discretize(dur):
            return np.digitize(dur, bins=self._time_bins) - 1

        if validation_data is not None:
            Xv, tv, ev = validation_data
            Xv, tv, ev = CoxTimeModel._to_dataframe(Xv, tv, ev)
            Xv = _encode_non_numeric(Xv)
            for c in self._feature_cols:
                if c not in Xv.columns:
                    Xv[c] = 0.0
            Xv = Xv[self._feature_cols]
            x_train = X.astype("float32").values
            x_val = Xv.astype("float32").values
            y_train = (_discretize(time_s.values).astype("int64"), event_s.astype("float32").values)
            y_val = (_discretize(tv.values).astype("int64"), ev.astype("float32").values)
        else:
            train_idx, val_idx = train_test_split(
                df.index, test_size=0.2, stratify=df["event"], random_state=42,
            )
            train_df = df.loc[train_idx]
            val_df = df.loc[val_idx]
            x_train = train_df[self._feature_cols].astype("float32").values
            x_val = val_df[self._feature_cols].astype("float32").values
            y_train = (_discretize(train_df["duration"].values).astype("int64"), train_df["event"].astype("float32").values)
            y_val = (_discretize(val_df["duration"].values).astype("int64"), val_df["event"].astype("float32").values)

        in_features = x_train.shape[1]
        net = tt.practical.MLPVanilla(
            in_features, self.num_nodes, self.out_features,
            batch_norm=self.batch_norm, dropout=self.dropout,
        )
        model = DeepHitSingle(net, tt.optim.Adam, alpha=self.alpha, sigma=self.sigma)
        model.optimizer.set_lr(learning_rate)

        callbacks = [tt.callbacks.EarlyStopping(patience=patience)] if early_stopping else []
        model.fit(
            x_train, y_train,
            batch_size, epochs, callbacks, bool(verbose),
            val_data=(x_val, y_val), val_batch_size=batch_size,
        )
        self._pycox_model = model
        self.is_fitted = True

    def predict_survival(self, X, times=None) -> np.ndarray:
        self._check_fitted()
        x = self._prepare_input(X)
        surv_df = self._pycox_model.predict_surv_df(x)
        surv = surv_df.values.T
        if times is not None:
            surv = CoxTimeModel._interpolate_surv(surv, surv_df.index.values, np.asarray(times))
        return surv

    def predict_risk(self, X) -> np.ndarray:
        surv = self.predict_survival(X)
        return 1.0 - surv.mean(axis=1)

    def evaluate(self, X, time, event) -> Dict[str, float]:
        _require_pycox()
        from pycox.evaluation import EvalSurv
        x = self._prepare_input(X)
        surv_df = self._pycox_model.predict_surv_df(x)
        durations_disc = np.digitize(np.asarray(time, dtype="float32"), bins=self._time_bins) - 1
        events = np.asarray(event, dtype="float32")
        ev = EvalSurv(surv_df, durations_disc, events, censor_surv="km")
        time_grid = np.linspace(
            max(1, float(np.asarray(time).min())),
            float(np.asarray(time).max()),
            100,
        )
        return {
            "c_index": ev.concordance_td(),
            "integrated_brier_score": ev.integrated_brier_score(time_grid),
            "integrated_nbll": ev.integrated_nbll(time_grid),
        }

    def _get_model_config(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
            "num_intervals": self.num_intervals,
            "out_features": self.out_features,
            "alpha": self.alpha,
            "sigma": self.sigma,
            "feature_cols": self._feature_cols,
        }

    def _create_model_from_config(self, config: Dict[str, Any]) -> None:
        self.num_nodes = config.get("num_nodes", [32, 32])
        self.batch_norm = config.get("batch_norm", True)
        self.dropout = config.get("dropout", 0.01)
        self.num_intervals = config.get("num_intervals", 20)
        self.out_features = config.get("out_features", 4000)
        self.alpha = config.get("alpha", 0.2)
        self.sigma = config.get("sigma", 0.1)
        self._feature_cols = config.get("feature_cols", [])

    def _check_fitted(self):
        if not self.is_fitted or self._pycox_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def _prepare_input(self, X) -> np.ndarray:
        return CoxTimeModel._prepare_input(self, X)
