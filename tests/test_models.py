# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Tests for TexGISa survival model
"""

import numpy as np
import pytest
import torch

from texgisa_survival import TexGISa


@pytest.fixture
def small_survival_data():
    """Generate small synthetic survival dataset for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    time = np.random.exponential(50, n_samples).astype(np.float32)
    event = np.random.binomial(1, 0.7, n_samples).astype(np.float32)

    return {'X': X, 'y': time, 'e': event, 'time': time, 'event': event}


class TestTexGISa:
    """Test suite for TexGISa model"""

    def test_initialization(self):
        """Test model initialization"""
        model = TexGISa(hidden_layers=[64, 32], num_time_bins=10)
        assert model is not None
        assert model.model.hidden_layers == [64, 32]
        assert model.model.num_time_bins == 10

    def test_initialization_with_defaults(self):
        """Test model initialization with default parameters"""
        model = TexGISa()
        assert model is not None
        assert model.model.hidden_layers == [128, 64, 32]  # Default
        assert model.model.num_time_bins == 20  # Default

    def test_fit(self, small_survival_data):
        """Test model fitting"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)
        model.fit(X, time, event, epochs=5, verbose=0)

        # Check that model was trained
        assert model.is_fitted == True
        assert hasattr(model, 'model')

    def test_predict_risk(self, small_survival_data):
        """Test risk prediction"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)
        model.fit(X, time, event, epochs=5, verbose=0)

        risk_scores = model.predict_risk(X)
        assert risk_scores is not None
        assert len(risk_scores) == len(X)
        assert np.all(np.isfinite(risk_scores))
        assert risk_scores.dtype == np.float32 or risk_scores.dtype == np.float64

    def test_predict_survival(self, small_survival_data):
        """Test survival probability prediction"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)
        model.fit(X, time, event, epochs=5, verbose=0)

        # Predict at all time bins
        survival_probs = model.predict_survival(X)
        assert survival_probs is not None
        assert survival_probs.shape == (len(X), 5)
        assert np.all(survival_probs >= 0)
        assert np.all(survival_probs <= 1)

    def test_predict_survival_at_times(self, small_survival_data):
        """Test survival probability prediction at specific times"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=10)
        model.fit(X, time, event, epochs=5, verbose=0)

        # Predict at specific times
        test_times = np.array([10.0, 30.0, 50.0])
        survival_probs = model.predict_survival(X, times=test_times)
        assert survival_probs is not None
        assert survival_probs.shape == (len(X), len(test_times))
        assert np.all(survival_probs >= 0)
        assert np.all(survival_probs <= 1)

    def test_get_feature_importance(self, small_survival_data):
        """Test TEXGI feature importance computation"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5, ig_steps=10)
        model.fit(X, time, event, epochs=5, verbose=0)

        importance_df = model.get_feature_importance(method='texgi')

        # Check structure (should be a DataFrame)
        assert importance_df is not None
        assert 'importance' in importance_df.columns
        assert 'std' in importance_df.columns
        assert 'feature' in importance_df.columns

        # Check importance scores
        assert len(importance_df) == X.shape[1]
        assert np.all(np.isfinite(importance_df['importance']))
        assert np.all(importance_df['importance'] >= 0)

        # Check normalization (should sum to ~1)
        assert np.abs(importance_df['importance'].sum() - 1.0) < 0.01

    def test_expert_rules(self, small_survival_data):
        """Test expert knowledge integration"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(
            hidden_layers=[32, 16],
            num_time_bins=5,
            lambda_expert=0.1
        )

        # Add expert rules
        model.add_expert_rule(feature=0, relation='>', threshold='mean', sign=1, weight=1.0)
        model.add_expert_rule(feature=1, relation='<', threshold='median', sign=-1, weight=0.5)

        # Check rules were added
        assert len(model.expert_rules) == 2
        assert model.expert_rules[0]['feature'] == 0
        assert model.expert_rules[0]['sign'] == 1
        assert model.expert_rules[1]['feature'] == 1
        assert model.expert_rules[1]['sign'] == -1

        # Train with rules
        model.fit(X, time, event, epochs=5, verbose=0)
        assert model.is_fitted == True

    def test_different_architectures(self, small_survival_data):
        """Test model with different architectures"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        architectures = [
            [32],
            [64, 32],
            [128, 64, 32]
        ]

        for hidden_layers in architectures:
            model = TexGISa(hidden_layers=hidden_layers, num_time_bins=5)
            model.fit(X, time, event, epochs=3, verbose=0)

            risk = model.predict_risk(X)
            assert risk is not None
            assert len(risk) == len(X)

    def test_different_time_bins(self, small_survival_data):
        """Test model with different numbers of time bins"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        time_bins_list = [5, 10, 15]

        for num_bins in time_bins_list:
            model = TexGISa(hidden_layers=[32, 16], num_time_bins=num_bins)
            model.fit(X, time, event, epochs=3, verbose=0)

            survival = model.predict_survival(X)
            assert survival.shape == (len(X), num_bins)

    def test_validation_data(self, small_survival_data):
        """Test training with validation data"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        # Split into train and validation
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        time_train, time_val = time[:split_idx], time[split_idx:]
        event_train, event_val = event[:split_idx], event[split_idx:]

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)
        model.fit(
            X_train, time_train, event_train,
            validation_data=(X_val, time_val, event_val),
            epochs=5,
            verbose=0
        )

        assert model.is_fitted == True

    def test_early_stopping(self, small_survival_data):
        """Test early stopping functionality"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        # Split into train and validation
        split_idx = len(X) // 2
        X_train, X_val = X[:split_idx], X[split_idx:]
        time_train, time_val = time[:split_idx], time[split_idx:]
        event_train, event_val = event[:split_idx], event[split_idx:]

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)
        model.fit(
            X_train, time_train, event_train,
            validation_data=(X_val, time_val, event_val),
            epochs=100,  # Many epochs
            early_stopping=True,
            patience=5,
            verbose=0
        )

        # Should stop early (not run all 100 epochs)
        assert model.is_fitted == True

    def test_regularization_parameters(self, small_survival_data):
        """Test different regularization parameters"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        # Test with temporal smoothness
        model1 = TexGISa(
            hidden_layers=[32, 16],
            num_time_bins=5,
            lambda_smooth=0.1
        )
        model1.fit(X, time, event, epochs=3, verbose=0)
        assert model1.is_fitted == True

        # Test with expert knowledge
        model2 = TexGISa(
            hidden_layers=[32, 16],
            num_time_bins=5,
            lambda_expert=0.1
        )
        model2.add_expert_rule(feature=0, relation='>', threshold='mean', sign=1)
        model2.fit(X, time, event, epochs=3, verbose=0)
        assert model2.is_fitted == True

    def test_device_cpu(self, small_survival_data):
        """Test model on CPU device"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5, device='cpu')
        model.fit(X, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True
        assert str(model.device) == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_cuda(self, small_survival_data):
        """Test model on CUDA device"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5, device='cuda')
        model.fit(X, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True
        assert 'cuda' in str(model.device)

    def test_random_state(self, small_survival_data):
        """Test reproducibility with random state"""
        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        # Train two models with same random state
        model1 = TexGISa(hidden_layers=[32, 16], num_time_bins=5, random_state=42)
        model1.fit(X, time, event, epochs=5, verbose=0)
        risk1 = model1.predict_risk(X)

        model2 = TexGISa(hidden_layers=[32, 16], num_time_bins=5, random_state=42)
        model2.fit(X, time, event, epochs=5, verbose=0)
        risk2 = model2.predict_risk(X)

        # Results should be similar (not exactly same due to TEXGI randomness)
        assert np.corrcoef(risk1, risk2)[0, 1] > 0.9

    def test_error_predict_before_fit(self, small_survival_data):
        """Test that prediction before fitting raises error"""
        X = small_survival_data['X']

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)

        with pytest.raises(ValueError):
            model.predict_risk(X)

        with pytest.raises(ValueError):
            model.predict_survival(X)

    def test_error_importance_before_fit(self):
        """Test that importance computation before fitting raises error"""
        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)

        with pytest.raises(ValueError):
            model.get_feature_importance(method='texgi')

    def test_feature_names(self, small_survival_data):
        """Test with feature names"""
        import pandas as pd

        X = small_survival_data['X']
        time = small_survival_data['time']
        event = small_survival_data['event']

        # Create DataFrame with feature names
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=feature_names)

        model = TexGISa(hidden_layers=[32, 16], num_time_bins=5)

        # Add rule with feature name
        model.add_expert_rule(feature='feature_0', relation='>', threshold='mean', sign=1)

        model.fit(X_df, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True
        assert model.feature_names == feature_names


class TestTexGISaEdgeCases:
    """Test edge cases and error handling"""

    def test_small_dataset(self):
        """Test with very small dataset"""
        np.random.seed(42)
        X = np.random.randn(20, 5).astype(np.float32)
        time = np.random.exponential(10, 20).astype(np.float32)
        event = np.random.binomial(1, 0.5, 20).astype(np.float32)

        model = TexGISa(hidden_layers=[16], num_time_bins=3)
        model.fit(X, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True

    def test_all_censored(self):
        """Test with all censored data"""
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        time = np.random.exponential(10, 50).astype(np.float32)
        event = np.zeros(50).astype(np.float32)  # All censored

        model = TexGISa(hidden_layers=[16], num_time_bins=5)
        model.fit(X, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True

    def test_all_events(self):
        """Test with all events (no censoring)"""
        np.random.seed(42)
        X = np.random.randn(50, 5).astype(np.float32)
        time = np.random.exponential(10, 50).astype(np.float32)
        event = np.ones(50).astype(np.float32)  # All events

        model = TexGISa(hidden_layers=[16], num_time_bins=5)
        model.fit(X, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True

    def test_constant_features(self):
        """Test with constant features"""
        np.random.seed(42)
        X = np.ones((50, 5)).astype(np.float32)  # Constant features
        time = np.random.exponential(10, 50).astype(np.float32)
        event = np.random.binomial(1, 0.5, 50).astype(np.float32)

        model = TexGISa(hidden_layers=[16], num_time_bins=5)
        model.fit(X, time, event, epochs=3, verbose=0)

        assert model.is_fitted == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
