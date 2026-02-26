# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""Tests for kaplan_meier_estimator."""

import numpy as np
import pytest

from texgisa_survival.metrics import kaplan_meier_estimator


class TestKaplanMeierEstimator:
    def test_basic_curve(self):
        durations = np.array([1, 2, 3, 4, 5])
        events = np.array([1, 0, 1, 0, 1])
        t, S = kaplan_meier_estimator(durations, events)
        assert t[0] == 0.0
        assert S[0] == 1.0
        # Should have decreasing survival at event times
        assert len(t) >= 2
        assert S[-1] < 1.0

    def test_all_censored(self):
        durations = np.array([1, 2, 3])
        events = np.array([0, 0, 0])
        t, S = kaplan_meier_estimator(durations, events)
        # No events: survival stays at 1.0
        assert len(t) == 1
        assert S[0] == 1.0

    def test_all_events(self):
        durations = np.array([1, 2, 3, 4, 5])
        events = np.array([1, 1, 1, 1, 1])
        t, S = kaplan_meier_estimator(durations, events)
        # Should eventually reach 0
        assert S[-1] == 0.0

    def test_monotonicity(self):
        rng = np.random.RandomState(123)
        durations = rng.exponential(10, 100)
        events = rng.binomial(1, 0.5, 100)
        t, S = kaplan_meier_estimator(durations, events)
        # Survival must be non-increasing
        assert np.all(np.diff(S) <= 1e-10)

    def test_bin_width(self):
        durations = np.array([1.1, 1.9, 2.5, 3.3, 4.7])
        events = np.array([1, 1, 1, 0, 1])
        t1, S1 = kaplan_meier_estimator(durations, events, bin_width=0)
        t2, S2 = kaplan_meier_estimator(durations, events, bin_width=2)
        # Binning should produce fewer unique time points
        assert len(t2) <= len(t1)

    def test_limit_to_last_event(self):
        durations = np.array([1, 2, 3, 10])
        events = np.array([1, 1, 0, 0])
        t_lim, S_lim = kaplan_meier_estimator(durations, events, limit_to_last_event=True)
        t_all, S_all = kaplan_meier_estimator(durations, events, limit_to_last_event=False)
        # With limit, late censored observations (time=10) are excluded
        assert t_lim[-1] <= 2.0
        # Without limit, they can appear in the at-risk set

    def test_returns_numpy_arrays(self):
        t, S = kaplan_meier_estimator([1, 2, 3], [1, 0, 1])
        assert isinstance(t, np.ndarray)
        assert isinstance(S, np.ndarray)

    def test_starts_at_zero_one(self):
        t, S = kaplan_meier_estimator([5, 10, 15], [1, 1, 0])
        assert t[0] == 0.0
        assert S[0] == 1.0
