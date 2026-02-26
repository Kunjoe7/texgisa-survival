# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
End-to-end integration test for texgisa-survival v2.0.0.

Validates the full pipeline: DataManager → KM curve → baseline model →
counterfactual explanations, using synthetic data.
"""

import numpy as np
import pandas as pd
import pytest
import torch

pycox = pytest.importorskip("pycox", reason="pycox not installed")
tt = pytest.importorskip("torchtuples", reason="torchtuples not installed")

from texgisa_survival import __version__
from texgisa_survival.data.manager import DataManager
from texgisa_survival.metrics import (
    concordance_index,
    kaplan_meier_estimator,
)
from texgisa_survival.models.baselines import CoxTimeModel
from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
from texgisa_survival.explainers.counterfactual import (
    CFResult,
    TexgiCFResult,
    CounterfactualExplainer,
    generate_cf_from_arrays,
    generate_texgi_counterfactuals,
)


@pytest.fixture
def e2e_data():
    """Synthetic survival dataset for e2e testing."""
    rng = np.random.RandomState(42)
    n, d = 200, 5
    X = rng.randn(n, d).astype("float32")
    time = rng.exponential(50, n).astype("float32") + 1.0
    event = rng.binomial(1, 0.6, n).astype("float32")
    feature_names = [f"feat_{i}" for i in range(d)]
    df = pd.DataFrame(X, columns=feature_names)
    df["duration"] = time
    df["event"] = event
    return {
        "X": X,
        "time": time,
        "event": event,
        "feature_names": feature_names,
        "df": df,
        "n": n,
        "d": d,
    }


class TestVersion:
    def test_version_is_2(self):
        assert __version__ == "2.0.0"


class TestDataManagerE2E:
    def test_load_and_summary(self, e2e_data):
        dm = DataManager()
        dm.load_data(e2e_data["df"], file_name="synthetic.csv")
        assert dm.get_data() is not None
        assert dm.get_current_file_name() == "synthetic.csv"

        summary = dm.get_data_summary()
        assert summary["num_rows"] == e2e_data["n"]
        assert summary["num_columns"] == e2e_data["d"] + 2  # features + duration + event
        assert "column_names" in summary
        assert "missing_values" in summary

    def test_multimodal_load(self, e2e_data):
        dm = DataManager()
        tabular = e2e_data["df"][e2e_data["feature_names"]]
        sensor = pd.DataFrame({"sensor_0": np.random.randn(e2e_data["n"])})
        dm.load_multimodal_data(tabular_df=tabular, sensor_df=sensor)
        assert dm.tabular_df is not None
        assert dm.sensor_df is not None
        assert dm.image_df is None

        summary = dm.get_data_summary()
        assert "modalities" in summary
        assert "tabular" in summary["modalities"]
        assert "sensor" in summary["modalities"]


class TestKaplanMeierE2E:
    def test_km_curve_properties(self, e2e_data):
        t, S = kaplan_meier_estimator(e2e_data["time"], e2e_data["event"])

        # Starts at (0, 1)
        assert t[0] == 0.0
        assert S[0] == 1.0

        # Monotonically non-increasing
        assert np.all(np.diff(S) <= 1e-10)

        # Probabilities in [0, 1]
        assert np.all(S >= 0.0)
        assert np.all(S <= 1.0)

        # Should have dropped below 1 with 60% event rate
        assert S[-1] < 1.0


@pytest.mark.slow
class TestBaselineModelE2E:
    def test_coxtime_full_pipeline(self, e2e_data):
        X, time, event = e2e_data["X"], e2e_data["time"], e2e_data["event"]

        # Train
        model = CoxTimeModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        assert model.is_fitted

        # predict_survival
        surv = model.predict_survival(X[:20])
        assert surv.shape[0] == 20
        assert surv.shape[1] > 0
        assert np.all(surv >= 0) and np.all(surv <= 1.01)

        # predict_risk
        risk = model.predict_risk(X[:20])
        assert risk.shape == (20,)

        # evaluate
        metrics = model.evaluate(X, time, event)
        assert "c_index" in metrics
        assert 0.0 <= metrics["c_index"] <= 1.0


@pytest.mark.slow
class TestCounterfactualScalingE2E:
    def test_scaling_from_hazards(self, e2e_data):
        hazards = np.random.RandomState(42).rand(10, 8).astype("float32")

        result = generate_cf_from_arrays(
            hazards,
            desired_extension=1.0,
            patient_indices=[0, 1, 2],
        )
        assert isinstance(result, CFResult)
        assert not result.table.empty
        assert "sample_id" in result.table.columns
        ids = result.table["sample_id"].unique()
        assert set(ids).issubset({0, 1, 2})

    def test_scaling_via_explainer(self, e2e_data):
        hazards = np.random.RandomState(42).rand(5, 10).astype("float32")
        explainer = CounterfactualExplainer()
        result = explainer.explain(hazards=hazards, method="scaling", desired_extension=1.5)
        assert isinstance(result, CFResult)
        assert not result.table.empty


@pytest.mark.slow
class TestCounterfactualGradientE2E:
    def test_gradient_with_model(self, e2e_data):
        model = MultiTaskSurvivalNet(
            input_dim=e2e_data["d"],
            hidden_layers=[16],
            num_time_bins=8,
            dropout=0.0,
        )
        model.eval()

        result = generate_texgi_counterfactuals(
            model,
            e2e_data["X"][:5],
            feature_names=e2e_data["feature_names"],
            desired_extension=1.0,
            steps=10,
            patient_indices=[0, 1],
        )
        assert isinstance(result, TexgiCFResult)
        assert not result.table.empty or result.summary.get("patient_notes")

    def test_gradient_via_explainer(self, e2e_data):
        model = MultiTaskSurvivalNet(
            input_dim=e2e_data["d"],
            hidden_layers=[16],
            num_time_bins=8,
            dropout=0.0,
        )
        model.eval()

        explainer = CounterfactualExplainer(
            model=model,
            feature_names=e2e_data["feature_names"],
        )
        result = explainer.explain(
            e2e_data["X"][:3],
            method="gradient",
            desired_extension=2.0,
            steps=10,
            patient_indices=[0],
            top_k=3,
        )
        assert isinstance(result, TexgiCFResult)


@pytest.mark.slow
class TestFullPipelineE2E:
    """End-to-end: DataManager → KM → Baseline → Counterfactual."""

    def test_complete_workflow(self, e2e_data):
        X, time, event = e2e_data["X"], e2e_data["time"], e2e_data["event"]

        # 1. DataManager
        dm = DataManager()
        dm.load_data(e2e_data["df"], file_name="pipeline_test.csv")
        summary = dm.get_data_summary()
        assert summary["num_rows"] == e2e_data["n"]

        # 2. Kaplan-Meier
        t_km, S_km = kaplan_meier_estimator(time, event)
        assert t_km[0] == 0.0 and S_km[0] == 1.0
        assert np.all(np.diff(S_km) <= 1e-10)

        # 3. Train baseline
        model = CoxTimeModel(num_nodes=[16, 16], random_state=42)
        model.fit(X, time, event, epochs=5, batch_size=64, patience=3)
        assert model.is_fitted
        metrics = model.evaluate(X, time, event)
        assert 0.0 <= metrics["c_index"] <= 1.0

        # 4. Counterfactual (scaling) from synthetic hazards
        hazards = np.random.RandomState(99).rand(5, 8).astype("float32")
        cf_result = generate_cf_from_arrays(
            hazards, desired_extension=1.0, patient_indices=[0, 1]
        )
        assert isinstance(cf_result, CFResult)
        assert not cf_result.table.empty

        # 5. Counterfactual (gradient) with TexGISa net
        net = MultiTaskSurvivalNet(
            input_dim=e2e_data["d"],
            hidden_layers=[16],
            num_time_bins=8,
            dropout=0.0,
        )
        net.eval()
        texgi_result = generate_texgi_counterfactuals(
            net,
            X[:3],
            feature_names=e2e_data["feature_names"],
            desired_extension=1.0,
            steps=10,
            patient_indices=[0],
        )
        assert isinstance(texgi_result, TexgiCFResult)
