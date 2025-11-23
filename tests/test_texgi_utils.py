# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Tests for TEXGI utility functions
"""

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from texgisa_survival.texgi_utils import (
    sample_extreme_code,
    attribution_temporal_l1,
    aggregate_importance,
    expert_penalty,
    resolve_important_feature_indices,
    masked_bce_nll,
    topk_feature_importance,
    integrated_gradients_time,
    texgi_time_series,
    standardize_features,
    destandardize_features,
)


class SimpleHazardModel(nn.Module):
    """Simple model for testing that outputs hazards."""

    def __init__(self, input_dim: int, num_bins: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_bins)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class TestSampleExtremeCode:
    """Test suite for GPD sampling"""

    def test_shape(self):
        """Test output shape"""
        batch_size = 32
        extreme_dim = 2
        e = sample_extreme_code(batch_size, extreme_dim)

        assert e.shape == (batch_size, extreme_dim)

    def test_positive_values(self):
        """Test that GPD samples are positive"""
        e = sample_extreme_code(100, 1)
        assert torch.all(e >= 0)

    def test_device(self):
        """Test device placement"""
        e = sample_extreme_code(10, 1, device="cpu")
        assert e.device.type == "cpu"

    def test_different_parameters(self):
        """Test with different GPD parameters"""
        e1 = sample_extreme_code(100, 1, xi=0.1, beta=0.5)
        e2 = sample_extreme_code(100, 1, xi=0.5, beta=2.0)

        # Different parameters should give different distributions
        assert e1.mean() != e2.mean()


class TestAttributionTemporalL1:
    """Test suite for temporal smoothness penalty"""

    def test_zero_penalty_for_constant(self):
        """Test that constant attributions give zero penalty"""
        phi = torch.ones(5, 10, 8)  # T=5, B=10, D=8
        penalty = attribution_temporal_l1(phi)

        assert penalty.item() == pytest.approx(0.0)

    def test_nonzero_penalty_for_varying(self):
        """Test that varying attributions give nonzero penalty"""
        phi = torch.randn(5, 10, 8)
        penalty = attribution_temporal_l1(phi)

        assert penalty.item() > 0

    def test_single_time_step(self):
        """Test with single time step"""
        phi = torch.randn(1, 10, 8)
        penalty = attribution_temporal_l1(phi)

        assert penalty.item() == 0.0

    def test_invalid_dimensions(self):
        """Test error for invalid dimensions"""
        phi = torch.randn(10, 8)  # Missing batch dimension
        with pytest.raises(ValueError):
            attribution_temporal_l1(phi)


class TestAggregateImportance:
    """Test suite for importance aggregation"""

    def test_output_shapes(self):
        """Test output shapes"""
        T, B, D = 5, 10, 8
        phi = torch.randn(T, B, D)

        imp_abs, imp_dir = aggregate_importance(phi)

        assert imp_abs.shape == (D,)
        assert imp_dir.shape == (D,)

    def test_absolute_nonnegative(self):
        """Test that absolute importance is non-negative"""
        phi = torch.randn(5, 10, 8)
        imp_abs, _ = aggregate_importance(phi)

        assert torch.all(imp_abs >= 0)

    def test_directional_preserves_sign(self):
        """Test that directional importance preserves sign"""
        phi = torch.ones(5, 10, 8) * -1.0  # All negative
        _, imp_dir = aggregate_importance(phi)

        assert torch.all(imp_dir < 0)


class TestExpertPenalty:
    """Test suite for expert prior penalty"""

    def test_zero_importance_zero_penalty(self):
        """Test with no important features"""
        phi = torch.randn(5, 10, 8)
        penalty = expert_penalty(phi, [])

        # Should still penalize non-important features
        assert penalty.item() >= 0

    def test_with_important_features(self):
        """Test with important features"""
        phi = torch.randn(5, 10, 8)
        penalty = expert_penalty(phi, [0, 1, 2])

        assert penalty.item() >= 0

    def test_invalid_dimensions(self):
        """Test error for invalid dimensions"""
        phi = torch.randn(10, 8)
        with pytest.raises(ValueError):
            expert_penalty(phi, [0])


class TestResolveImportantFeatureIndices:
    """Test suite for resolving important feature indices"""

    def test_from_list(self):
        """Test extraction from list"""
        config = {"important_features": ["feat_0", "feat_2"]}
        feat2idx = {"feat_0": 0, "feat_1": 1, "feat_2": 2}

        indices = resolve_important_feature_indices(config, feat2idx)

        assert indices == [0, 2]

    def test_from_rules(self):
        """Test extraction from rules"""
        config = {
            "rules": [
                {"feature": "feat_1", "relation": ">=mean"},
                {"feature": "feat_3", "important": True},
            ]
        }
        feat2idx = {"feat_0": 0, "feat_1": 1, "feat_2": 2, "feat_3": 3}

        indices = resolve_important_feature_indices(config, feat2idx)

        assert 1 in indices
        assert 3 in indices

    def test_empty_config(self):
        """Test with empty config"""
        indices = resolve_important_feature_indices(None, {})
        assert indices == []


class TestMaskedBceNll:
    """Test suite for masked BCE loss"""

    def test_basic_loss(self):
        """Test basic loss computation"""
        hazards = torch.tensor([[0.3, 0.5, 0.7], [0.2, 0.4, 0.6]])
        labels = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=torch.float32)
        masks = torch.ones_like(labels)

        loss = masked_bce_nll(hazards, labels, masks)

        assert loss.item() > 0
        assert torch.isfinite(loss)

    def test_with_masks(self):
        """Test that masks are respected"""
        hazards = torch.tensor([[0.9, 0.9, 0.9]])  # All high
        labels = torch.tensor([[0, 0, 1]], dtype=torch.float32)  # Only last is event
        masks = torch.tensor([[0, 0, 1]], dtype=torch.float32)  # Only last is valid

        loss = masked_bce_nll(hazards, labels, masks)

        # Loss should be low since prediction matches label in valid position
        assert loss.item() < 1.0

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        hazards = torch.tensor([[0.0001, 0.9999]])
        labels = torch.tensor([[1, 0]], dtype=torch.float32)
        masks = torch.ones_like(labels)

        loss = masked_bce_nll(hazards, labels, masks)

        assert torch.isfinite(loss)


class TestTopkFeatureImportance:
    """Test suite for top-k feature importance"""

    def test_returns_k_features(self):
        """Test that k features are returned"""
        phi = torch.randn(5, 10, 8)
        feature_names = [f"feat_{i}" for i in range(8)]

        df = topk_feature_importance(phi, feature_names, k=3)

        assert len(df) == 3
        assert "feature" in df.columns
        assert "importance" in df.columns

    def test_sorted_by_importance(self):
        """Test that results are sorted by importance"""
        phi = torch.randn(5, 10, 8)
        feature_names = [f"feat_{i}" for i in range(8)]

        df = topk_feature_importance(phi, feature_names, k=5)

        assert df["importance"].is_monotonic_decreasing


class TestIntegratedGradientsTime:
    """Test suite for time-specific integrated gradients"""

    def test_output_shape(self):
        """Test output shape"""
        model = SimpleHazardModel(10, 5)
        X = torch.randn(8, 10)
        X_baseline = torch.randn(8, 10)

        attributions = integrated_gradients_time(model, X, X_baseline, hazard_index=2)

        assert attributions.shape == X.shape

    def test_baseline_attribution_zero(self):
        """Test that attribution at baseline is zero"""
        model = SimpleHazardModel(10, 5)
        X_baseline = torch.randn(8, 10)

        attributions = integrated_gradients_time(model, X_baseline, X_baseline, hazard_index=0)

        # Attributions should be close to zero when input equals baseline
        assert torch.allclose(attributions, torch.zeros_like(attributions), atol=1e-5)


class TestTexgiTimeSeries:
    """Test suite for time-series TEXGI"""

    def test_output_shape(self):
        """Test output shape"""
        model = SimpleHazardModel(10, 5)
        X = torch.randn(8, 10)

        phi = texgi_time_series(model, X, M=5)

        assert phi.shape[0] == 5  # T time bins
        assert phi.shape[1] == 8  # B batch size
        assert phi.shape[2] == 10  # D features

    def test_with_baseline(self):
        """Test with explicit baseline"""
        model = SimpleHazardModel(10, 5)
        X = torch.randn(8, 10)
        X_baseline = torch.zeros(8, 10)

        phi = texgi_time_series(model, X, X_baseline=X_baseline, M=5)

        assert phi.shape == (5, 8, 10)

    def test_with_time_sampling(self):
        """Test with time bin sampling"""
        model = SimpleHazardModel(10, 20)  # 20 time bins
        X = torch.randn(8, 10)

        phi = texgi_time_series(model, X, M=5, t_sample=5)

        # Should only compute for 5 time bins
        assert phi.shape[0] <= 5


class TestStandardizeFeatures:
    """Test suite for feature standardization"""

    def test_standardized_stats(self):
        """Test that standardized features have correct stats"""
        X = torch.randn(100, 10) * 5 + 10  # Mean ~10, std ~5

        X_std, mu, std = standardize_features(X)

        # Standardized should have mean ~0 and std ~1
        assert X_std.mean().abs() < 0.1
        assert (X_std.std() - 1.0).abs() < 0.1

    def test_roundtrip(self):
        """Test standardize and destandardize roundtrip"""
        X = torch.randn(50, 8)

        X_std, mu, std = standardize_features(X)
        X_recovered = destandardize_features(X_std, mu, std)

        assert torch.allclose(X, X_recovered, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
