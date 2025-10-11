# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Tests for evaluation metrics
"""

import numpy as np
import pytest

from texgisa_survival.metrics import (
    concordance_index,
    brier_score,
    integrated_brier_score,
    cumulative_dynamic_auc
)


class TestConcordanceIndex:
    """Test suite for concordance index"""
    
    def test_perfect_concordance(self):
        """Test perfect concordance"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        e = np.ones(5)
        
        c_index = concordance_index(y_true, y_pred, e)
        assert c_index == 1.0
    
    def test_reverse_concordance(self):
        """Test reverse concordance"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([5, 4, 3, 2, 1])
        e = np.ones(5)
        
        c_index = concordance_index(y_true, y_pred, e)
        assert c_index == 0.0
    
    def test_random_concordance(self):
        """Test random concordance (should be around 0.5)"""
        np.random.seed(42)
        n = 1000
        y_true = np.random.exponential(1, n)
        y_pred = np.random.randn(n)
        e = np.random.binomial(1, 0.7, n)
        
        c_index = concordance_index(y_true, y_pred, e)
        assert 0.4 <= c_index <= 0.6
    
    def test_with_censoring(self):
        """Test concordance with censored data"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        e = np.array([1, 0, 1, 0, 1])
        
        c_index = concordance_index(y_true, y_pred, e)
        assert 0 <= c_index <= 1
    
    def test_invalid_input(self):
        """Test with invalid input"""
        with pytest.raises(ValueError):
            concordance_index([1, 2], [1, 2, 3], [1, 1])


class TestBrierScore:
    """Test suite for Brier score"""
    
    def test_perfect_prediction(self):
        """Test perfect prediction (Brier score = 0)"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 1, 1, 1, 1])
        # Perfect predictions: 1 for events before time, 0 after
        y_pred_at_25 = np.array([1, 1, 0, 0, 0])
        
        score = brier_score(y_true, y_pred_at_25, e, time=25)
        assert score == pytest.approx(0, abs=0.1)
    
    def test_worst_prediction(self):
        """Test worst prediction (Brier score = 1)"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 1, 1, 1, 1])
        # Worst predictions: opposite of truth
        y_pred_at_25 = np.array([0, 0, 1, 1, 1])
        
        score = brier_score(y_true, y_pred_at_25, e, time=25)
        assert score > 0.5
    
    def test_probabilistic_predictions(self):
        """Test with probabilistic predictions"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        
        score = brier_score(y_true, y_pred, e, time=25)
        assert 0 <= score <= 1
    
    def test_with_censoring(self):
        """Test Brier score with censored data"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
        
        score = brier_score(y_true, y_pred, e, time=25)
        assert 0 <= score <= 1


class TestIntegratedBrierScore:
    """Test suite for integrated Brier score"""
    
    def test_integrated_score(self):
        """Test integrated Brier score calculation"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 1, 1, 1, 1])
        times = np.array([15, 25, 35, 45])
        
        # Predictions at different times
        y_pred = np.array([
            [0.9, 0.7, 0.5, 0.3],  # Sample 1
            [0.8, 0.6, 0.4, 0.2],  # Sample 2
            [0.7, 0.5, 0.3, 0.1],  # Sample 3
            [0.6, 0.4, 0.2, 0.1],  # Sample 4
            [0.5, 0.3, 0.1, 0.0],  # Sample 5
        ])
        
        ibs = integrated_brier_score(y_true, y_pred, e, times)
        assert 0 <= ibs <= 1
    
    def test_single_time_point(self):
        """Test with single time point (should equal Brier score)"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 1, 1, 1, 1])
        times = np.array([25])
        y_pred = np.array([[0.9], [0.7], [0.5], [0.3], [0.1]])
        
        ibs = integrated_brier_score(y_true, y_pred, e, times)
        bs = brier_score(y_true, y_pred[:, 0], e, time=25)
        
        assert ibs == pytest.approx(bs, abs=0.1)


class TestCumulativeDynamicAUC:
    """Test suite for cumulative dynamic AUC"""
    
    def test_perfect_discrimination(self):
        """Test perfect discrimination (AUC = 1)"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 1, 1, 1, 1])
        # Perfect risk scores
        y_pred = np.array([5, 4, 3, 2, 1])
        times = np.array([25, 35])
        
        auc_scores = cumulative_dynamic_auc(y_true, y_pred, e, times)
        assert len(auc_scores) == len(times)
        assert all(0.8 <= auc <= 1.0 for auc in auc_scores)
    
    def test_random_discrimination(self):
        """Test random discrimination (AUC ≈ 0.5)"""
        np.random.seed(42)
        n = 100
        y_true = np.random.exponential(1, n)
        e = np.ones(n)
        y_pred = np.random.randn(n)
        times = np.array([0.5, 1.0, 1.5])
        
        auc_scores = cumulative_dynamic_auc(y_true, y_pred, e, times)
        assert len(auc_scores) == len(times)
        assert all(0.3 <= auc <= 0.7 for auc in auc_scores)
    
    def test_with_censoring(self):
        """Test AUC with censored data"""
        y_true = np.array([10, 20, 30, 40, 50])
        e = np.array([1, 0, 1, 0, 1])
        y_pred = np.array([5, 4, 3, 2, 1])
        times = np.array([25, 35])
        
        auc_scores = cumulative_dynamic_auc(y_true, y_pred, e, times)
        assert len(auc_scores) == len(times)
        assert all(0 <= auc <= 1 for auc in auc_scores)