# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""Tests for baseline survival models (CoxTime, DeepSurv, DeepHit)."""

import numpy as np
import pytest

pycox = pytest.importorskip("pycox", reason="pycox not installed")
tt = pytest.importorskip("torchtuples", reason="torchtuples not installed")

from texgisa_survival.models.baselines import CoxTimeModel, DeepSurvModel, DeepHitModel


def _synthetic_data(n=200, d=5, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype("float32")
    time = rng.exponential(50, n).astype("float32") + 1.0
    event = rng.binomial(1, 0.6, n).astype("float32")
    return X, time, event


# ----- CoxTimeModel -----

@pytest.mark.slow
class TestCoxTimeModel:
    def test_fit_predict(self):
        X, time, event = _synthetic_data()
        model = CoxTimeModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        assert model.is_fitted

        surv = model.predict_survival(X[:10])
        assert surv.shape[0] == 10
        assert surv.shape[1] > 0
        assert np.all(surv >= 0) and np.all(surv <= 1.01)

    def test_predict_risk(self):
        X, time, event = _synthetic_data()
        model = CoxTimeModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        risk = model.predict_risk(X[:10])
        assert risk.shape == (10,)

    def test_evaluate(self):
        X, time, event = _synthetic_data()
        model = CoxTimeModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        metrics = model.evaluate(X, time, event)
        assert "c_index" in metrics
        assert 0.0 <= metrics["c_index"] <= 1.0


# ----- DeepSurvModel -----

@pytest.mark.slow
class TestDeepSurvModel:
    def test_fit_predict(self):
        X, time, event = _synthetic_data()
        model = DeepSurvModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        assert model.is_fitted

        surv = model.predict_survival(X[:10])
        assert surv.shape[0] == 10
        assert surv.shape[1] > 0

    def test_predict_risk(self):
        X, time, event = _synthetic_data()
        model = DeepSurvModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        risk = model.predict_risk(X[:10])
        assert risk.shape == (10,)

    def test_evaluate(self):
        X, time, event = _synthetic_data()
        model = DeepSurvModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        metrics = model.evaluate(X, time, event)
        assert "c_index" in metrics


# ----- DeepHitModel -----

@pytest.mark.slow
class TestDeepHitModel:
    def test_fit_predict(self):
        X, time, event = _synthetic_data()
        model = DeepHitModel(num_nodes=[16, 16], out_features=50, random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        assert model.is_fitted

        surv = model.predict_survival(X[:10])
        assert surv.shape[0] == 10
        assert surv.shape[1] > 0

    def test_predict_risk(self):
        X, time, event = _synthetic_data()
        model = DeepHitModel(num_nodes=[16, 16], out_features=50, random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        risk = model.predict_risk(X[:10])
        assert risk.shape == (10,)

    def test_evaluate(self):
        X, time, event = _synthetic_data()
        model = DeepHitModel(num_nodes=[16, 16], out_features=50, random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        metrics = model.evaluate(X, time, event)
        assert "c_index" in metrics
