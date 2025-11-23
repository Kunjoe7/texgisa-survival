# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Tests for diagnostics module
"""

import numpy as np
import pytest

from texgisa_survival.diagnostics import (
    brier_score_discrete,
    calibration_by_quantile,
    hazards_to_survival,
    integrated_brier_score_discrete,
    integrated_nll_discrete,
)


class TestBrierScoreDiscrete:
    """Test suite for discrete Brier score"""

    def test_basic_calculation(self):
        """Test basic Brier score calculation"""
        np.random.seed(42)
        n_samples = 50
        n_bins = 10

        # Create synthetic hazards
        hazards = np.random.uniform(0.05, 0.15, (n_samples, n_bins)).astype(np.float32)
        intervals = np.random.randint(1, n_bins + 1, n_samples)
        events = np.random.binomial(1, 0.7, n_samples).astype(np.float32)

        bs = brier_score_discrete(hazards, intervals, events)

        assert bs is not None
        assert len(bs) == n_bins
        assert np.all(bs >= 0)
        assert np.all(bs <= 1)

    def test_perfect_prediction(self):
        """Test with perfect predictions"""
        n_samples = 10
        n_bins = 5

        # Create hazards that perfectly predict the events
        hazards = np.zeros((n_samples, n_bins), dtype=np.float32)
        intervals = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        events = np.ones(n_samples, dtype=np.float32)

        # Set high hazard at the event time
        for i in range(n_samples):
            hazards[i, intervals[i] - 1] = 0.9

        bs = brier_score_discrete(hazards, intervals, events)

        assert bs is not None
        assert len(bs) == n_bins

    def test_empty_bins(self):
        """Test handling of empty intervals"""
        hazards = np.random.uniform(0, 0.2, (10, 5)).astype(np.float32)
        intervals = np.zeros(10)  # All zero intervals (invalid)
        events = np.ones(10, dtype=np.float32)

        bs = brier_score_discrete(hazards, intervals, events)

        # Should handle gracefully
        assert bs is not None
        assert len(bs) == 5


class TestCalibrationByQuantile:
    """Test suite for calibration by quantile"""

    def test_basic_calibration(self):
        """Test basic calibration calculation"""
        np.random.seed(42)
        n_samples = 100
        n_bins = 10

        hazards = np.random.uniform(0.05, 0.15, (n_samples, n_bins)).astype(np.float32)
        intervals = np.random.randint(1, n_bins + 1, n_samples)
        events = np.random.binomial(1, 0.7, n_samples).astype(np.float32)

        k_idx = 5
        pred_means, obs_rates, group_sizes = calibration_by_quantile(
            hazards, intervals, events, k_idx, n_bins=10
        )

        assert len(pred_means) > 0
        assert len(obs_rates) == len(pred_means)
        assert len(group_sizes) == len(pred_means)
        assert np.all(pred_means >= 0) and np.all(pred_means <= 1)
        assert np.all(obs_rates >= 0) and np.all(obs_rates <= 1)

    def test_different_bin_counts(self):
        """Test with different numbers of calibration bins"""
        np.random.seed(42)
        n_samples = 100
        n_time_bins = 10

        hazards = np.random.uniform(0.05, 0.15, (n_samples, n_time_bins)).astype(np.float32)
        intervals = np.random.randint(1, n_time_bins + 1, n_samples)
        events = np.random.binomial(1, 0.7, n_samples).astype(np.float32)

        for n_calib_bins in [5, 10, 20]:
            pred_means, obs_rates, group_sizes = calibration_by_quantile(
                hazards, intervals, events, k_idx=5, n_bins=n_calib_bins
            )
            assert len(pred_means) <= n_calib_bins


class TestHazardsToSurvival:
    """Test suite for hazards to survival conversion"""

    def test_basic_conversion(self):
        """Test basic conversion"""
        hazards = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]], dtype=np.float32)
        survival = hazards_to_survival(hazards)

        assert survival.shape == hazards.shape
        assert np.all(survival >= 0)
        assert np.all(survival <= 1)

    def test_monotonically_decreasing(self):
        """Test that survival is monotonically decreasing"""
        hazards = np.array([[0.1, 0.2, 0.3, 0.2, 0.1]], dtype=np.float32)
        survival = hazards_to_survival(hazards)

        # Survival should be monotonically decreasing
        assert np.all(np.diff(survival, axis=1) <= 0)

    def test_zero_hazards(self):
        """Test with zero hazards"""
        hazards = np.zeros((5, 10), dtype=np.float32)
        survival = hazards_to_survival(hazards)

        # With zero hazards, survival should be 1 everywhere
        assert np.allclose(survival, 1.0)

    def test_full_hazards(self):
        """Test with hazard=1"""
        hazards = np.ones((5, 10), dtype=np.float32)
        survival = hazards_to_survival(hazards)

        # With hazard=1, survival should be 0 everywhere
        assert np.allclose(survival, 0.0)


class TestIntegratedBrierScoreDiscrete:
    """Test suite for integrated Brier score"""

    def test_basic_calculation(self):
        """Test basic IBS calculation"""
        np.random.seed(42)
        n_samples = 50
        n_bins = 10

        hazards = np.random.uniform(0.05, 0.15, (n_samples, n_bins)).astype(np.float32)

        # Create one-hot labels
        labels = np.zeros((n_samples, n_bins), dtype=np.float32)
        event_times = np.random.randint(0, n_bins, n_samples)
        has_event = np.random.binomial(1, 0.7, n_samples)
        for i in range(n_samples):
            if has_event[i]:
                labels[i, event_times[i]] = 1

        # Create masks
        masks = np.ones((n_samples, n_bins), dtype=np.float32)

        ibs = integrated_brier_score_discrete(hazards, labels, masks)

        assert isinstance(ibs, float)
        assert 0 <= ibs <= 1

    def test_perfect_prediction(self):
        """Test IBS with perfect predictions"""
        n_samples = 20
        n_bins = 5

        # Create labels with events at bin 2
        labels = np.zeros((n_samples, n_bins), dtype=np.float32)
        labels[:, 2] = 1

        # Create perfect hazards
        hazards = np.zeros((n_samples, n_bins), dtype=np.float32)
        hazards[:, 2] = 1.0  # High hazard at event time

        masks = np.ones((n_samples, n_bins), dtype=np.float32)

        ibs = integrated_brier_score_discrete(hazards, labels, masks)

        # Should be low for good predictions
        assert isinstance(ibs, float)


class TestIntegratedNLLDiscrete:
    """Test suite for integrated negative log-likelihood"""

    def test_basic_calculation(self):
        """Test basic INLL calculation"""
        np.random.seed(42)
        n_samples = 50
        n_bins = 10

        hazards = np.random.uniform(0.1, 0.9, (n_samples, n_bins)).astype(np.float32)

        # Create one-hot labels
        labels = np.zeros((n_samples, n_bins), dtype=np.float32)
        event_times = np.random.randint(0, n_bins, n_samples)
        has_event = np.random.binomial(1, 0.7, n_samples)
        for i in range(n_samples):
            if has_event[i]:
                labels[i, event_times[i]] = 1

        masks = np.ones((n_samples, n_bins), dtype=np.float32)

        inll = integrated_nll_discrete(hazards, labels, masks)

        assert isinstance(inll, float)
        assert inll >= 0  # NLL should be non-negative

    def test_with_partial_masks(self):
        """Test with partial masks"""
        n_samples = 20
        n_bins = 10

        hazards = np.random.uniform(0.1, 0.9, (n_samples, n_bins)).astype(np.float32)
        labels = np.zeros((n_samples, n_bins), dtype=np.float32)

        # Partial masks (only first half of time bins valid)
        masks = np.zeros((n_samples, n_bins), dtype=np.float32)
        masks[:, :5] = 1.0

        inll = integrated_nll_discrete(hazards, labels, masks)

        assert isinstance(inll, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
