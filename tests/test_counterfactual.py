# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""Tests for counterfactual explanation modules."""

import numpy as np
import pandas as pd
import pytest
import torch

from texgisa_survival.explainers.counterfactual import (
    CFResult,
    TexgiCFResult,
    CounterfactualExplainer,
    generate_cf_from_arrays,
    generate_texgi_counterfactuals,
    _binary_scale_search,
    _target_cumhaz,
)


# ---------------------------------------------------------------------------
# Hazard-scaling tests
# ---------------------------------------------------------------------------

class TestTargetCumhaz:
    def test_positive_extension(self):
        ch = _target_cumhaz(10.0, horizon=10, desired_extension=5.0)
        assert ch < 10.0
        assert ch == pytest.approx(10.0 * 10.0 / 15.0, abs=1e-6)

    def test_zero_extension(self):
        assert _target_cumhaz(5.0, horizon=10, desired_extension=0.0) == 5.0

    def test_negative_extension(self):
        assert _target_cumhaz(5.0, horizon=10, desired_extension=-1.0) == 5.0


class TestBinaryScaleSearch:
    def test_uniform_scaling(self):
        haz = np.array([0.2, 0.3, 0.5])
        target = 0.5
        result = _binary_scale_search(haz, target)
        assert result.sum() <= target * 1.01

    def test_interval_scaling(self):
        haz = np.array([0.2, 0.3, 0.5])
        target = 0.6
        result = _binary_scale_search(haz, target, interval=2)
        assert result.sum() <= target * 1.01

    def test_already_below_target(self):
        haz = np.array([0.1, 0.1])
        result = _binary_scale_search(haz, target_ch=1.0)
        np.testing.assert_array_equal(result, haz)


class TestGenerateCFFromArrays:
    def test_basic(self):
        hazards = np.random.rand(5, 10).astype("float32")
        result = generate_cf_from_arrays(hazards, desired_extension=1.0)
        assert isinstance(result, CFResult)
        assert not result.table.empty
        assert "sample_id" in result.table.columns

    def test_with_patient_indices(self):
        hazards = np.random.rand(10, 8).astype("float32")
        result = generate_cf_from_arrays(hazards, patient_indices=[0, 3, 7])
        ids = result.table["sample_id"].unique()
        assert set(ids).issubset({0, 3, 7})

    def test_with_feature_suggestions(self):
        hazards = np.random.rand(5, 10).astype("float32")
        feature_df = pd.DataFrame(np.random.randn(5, 3), columns=["a", "b", "c"])
        result = generate_cf_from_arrays(
            hazards,
            feature_df=feature_df,
            feature_names=["a", "b", "c"],
            patient_indices=[0, 1],
            suggestions_per_patient=2,
        )
        assert not result.table.empty

    def test_torch_input(self):
        hazards = torch.rand(4, 6)
        result = generate_cf_from_arrays(hazards)
        assert isinstance(result, CFResult)
        assert len(result.table) >= 4


# ---------------------------------------------------------------------------
# Gradient-based tests
# ---------------------------------------------------------------------------

class TestGenerateTexgiCounterfactuals:
    @staticmethod
    def _make_model(input_dim=4, num_bins=8):
        from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
        model = MultiTaskSurvivalNet(
            input_dim=input_dim,
            hidden_layers=[16],
            num_time_bins=num_bins,
            dropout=0.0,
        )
        model.eval()
        return model

    def test_basic(self):
        model = self._make_model()
        features = np.random.randn(5, 4).astype("float32")
        result = generate_texgi_counterfactuals(
            model,
            features,
            feature_names=["f0", "f1", "f2", "f3"],
            desired_extension=1.0,
            steps=10,
            patient_indices=[0, 1],
        )
        assert isinstance(result, TexgiCFResult)
        # Should have at least some suggestions or patient notes
        assert not result.table.empty or result.summary.get("patient_notes")

    def test_immutable_features(self):
        model = self._make_model()
        features = np.random.randn(3, 4).astype("float32")
        result = generate_texgi_counterfactuals(
            model,
            features,
            feature_names=["f0", "f1", "f2", "f3"],
            immutable_features=["f0", "f1"],
            desired_extension=1.0,
            steps=10,
            patient_indices=[0],
        )
        if not result.table.empty:
            assert "f0" not in result.table["feature"].values
            assert "f1" not in result.table["feature"].values

    def test_with_dict_spec(self):
        result = generate_texgi_counterfactuals(
            {"input_dim": 4, "num_bins": 8, "hidden": 16, "depth": 1, "dropout": 0.0,
             "feature_names": ["a", "b", "c", "d"]},
            np.random.randn(3, 4).astype("float32"),
            desired_extension=1.0,
            steps=5,
            patient_indices=[0],
        )
        assert isinstance(result, TexgiCFResult)


# ---------------------------------------------------------------------------
# CounterfactualExplainer wrapper
# ---------------------------------------------------------------------------

class TestCounterfactualExplainer:
    @staticmethod
    def _make_model(input_dim=4, num_bins=8):
        from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
        return MultiTaskSurvivalNet(input_dim=input_dim, hidden_layers=[16], num_time_bins=num_bins, dropout=0.0)

    def test_scaling_method(self):
        hazards = np.random.rand(5, 10).astype("float32")
        explainer = CounterfactualExplainer()
        result = explainer.explain(hazards=hazards, method="scaling")
        assert isinstance(result, CFResult)

    def test_gradient_method(self):
        model = self._make_model()
        features = np.random.randn(3, 4).astype("float32")
        explainer = CounterfactualExplainer(model=model, feature_names=["a", "b", "c", "d"])
        result = explainer.explain(features, method="gradient", desired_extension=1.0, steps=5, patient_indices=[0])
        assert isinstance(result, TexgiCFResult)

    def test_invalid_method_raises(self):
        explainer = CounterfactualExplainer()
        with pytest.raises(ValueError, match="Unknown method"):
            explainer.explain(hazards=np.ones((2, 3)), method="invalid")
