# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Pytest configuration and fixtures for DHAI Survival tests
"""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def random_seed():
    """Fixed random seed for reproducible tests"""
    return 42


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds(random_seed):
    """Set random seeds for all libraries"""
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)


@pytest.fixture
def synthetic_survival_data():
    """Generate synthetic survival data for testing"""
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate survival times (exponential distribution)
    true_beta = np.random.randn(n_features)
    linear_predictor = X @ true_beta
    baseline_hazard = 0.01
    survival_times = np.random.exponential(
        1 / (baseline_hazard * np.exp(linear_predictor))
    )
    
    # Generate censoring times
    censoring_times = np.random.exponential(np.median(survival_times) * 1.5, n_samples)
    
    # Observed times and events
    y = np.minimum(survival_times, censoring_times)
    e = (survival_times <= censoring_times).astype(int)
    
    return {
        'X': X,
        'y': y,
        'e': e,
        'n_samples': n_samples,
        'n_features': n_features
    }


@pytest.fixture
def train_test_data(synthetic_survival_data):
    """Split synthetic data into train and test sets"""
    X = synthetic_survival_data['X']
    y = synthetic_survival_data['y']
    e = synthetic_survival_data['e']
    
    X_train, X_test, y_train, y_test, e_train, e_test = train_test_split(
        X, y, e, test_size=0.2, random_state=42
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'e_train': e_train,
        'e_test': e_test
    }


@pytest.fixture
def small_survival_data():
    """Small dataset for quick tests"""
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0]
    ])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e = np.array([1, 1, 0, 1, 0])
    
    return {'X': X, 'y': y, 'e': e}


@pytest.fixture
def device():
    """Get the appropriate device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")