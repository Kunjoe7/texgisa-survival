"""
Simple test to verify that the advanced functions work correctly.
"""

import sys
import torch
import numpy as np

# Add src to path
sys.path.append('src')

from texgisa_survival.models.texgisa_advanced import (
    TabularGeneratorWithRealInput,
    sample_extreme_code,
    integrated_gradients_time,
    texgi_time_series,
    attribution_temporal_l1,
    expert_penalty,
    _resolve_important_feature_indices,
    make_intervals,
    build_supervision,
    MySATrainer,
)

def test_gpd_sampling():
    """Test Generalized Pareto Distribution sampling."""
    print("Testing GPD sampling...")
    codes = sample_extreme_code(batch_size=100, extreme_dim=1, device='cpu', xi=0.3, beta=1.0)
    assert codes.shape == (100, 1)
    assert codes.min() >= 0  # GPD produces positive values
    print(f"  ✓ GPD sampling works. Mean: {codes.mean():.4f}, Std: {codes.std():.4f}")


def test_integrated_gradients():
    """Test integrated gradients computation."""
    print("\nTesting integrated gradients...")

    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
        torch.nn.Sigmoid()
    )
    model.eval()

    # Test data
    X = torch.randn(4, 10)
    X_baseline = torch.zeros(4, 10)

    # Compute IG for first time bin
    ig = integrated_gradients_time(model, X, X_baseline, hazard_index=0, M=10)
    assert ig.shape == X.shape
    print(f"  ✓ IG computation works. Attribution shape: {ig.shape}")


def test_texgi_time_series():
    """Test TEXGI time series computation."""
    print("\nTesting TEXGI time series...")

    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
        torch.nn.Sigmoid()
    )
    model.eval()

    # Test data
    X = torch.randn(4, 10)

    # Without generator (uses fallback baseline)
    phi = texgi_time_series(model, X, G=None, ref_stats={}, M=5, t_sample=3)
    assert phi.shape[0] == 3  # Time bins subsampled
    assert phi.shape[1] == 4  # Batch size
    assert phi.shape[2] == 10  # Features
    print(f"  ✓ TEXGI time series works. Attribution shape: {phi.shape}")


def test_temporal_smoothness():
    """Test temporal smoothness calculation."""
    print("\nTesting temporal smoothness...")

    # Create attribution tensor
    phi = torch.randn(5, 4, 10)  # [T, B, D]
    smooth = attribution_temporal_l1(phi)
    assert smooth.dim() == 0  # Scalar
    assert smooth >= 0
    print(f"  ✓ Temporal smoothness works. Value: {smooth.item():.4f}")


def test_expert_penalty():
    """Test expert penalty calculation."""
    print("\nTesting expert penalty...")

    # Create attribution tensor
    phi = torch.randn(5, 4, 10)  # [T, B, D]
    important_idx = [0, 1, 2]  # First 3 features are important

    penalty = expert_penalty(phi, important_idx)
    assert penalty.dim() == 0  # Scalar
    assert penalty >= 0
    print(f"  ✓ Expert penalty works. Value: {penalty.item():.4f}")


def test_data_processing():
    """Test data processing utilities."""
    print("\nTesting data processing...")

    # Create test dataframe
    import pandas as pd
    df = pd.DataFrame({
        'duration': np.random.exponential(10, 100),
        'event': np.random.binomial(1, 0.5, 100),
        'feat1': np.random.randn(100),
        'feat2': np.random.randn(100)
    })

    # Test make_intervals
    df = make_intervals(df, n_bins=10, method='quantile')
    assert 'interval_number' in df.columns
    print(f"  ✓ make_intervals works. Max interval: {df['interval_number'].max()}")

    # Test build_supervision
    labels, masks = build_supervision(
        df['interval_number'].values,
        df['event'].values,
        num_bins=10
    )
    assert labels.shape == (100, 10)
    assert masks.shape == (100, 10)
    print(f"  ✓ build_supervision works. Label shape: {labels.shape}")


def test_resolve_important_features():
    """Test important feature resolution."""
    print("\nTesting important feature resolution...")

    feat2idx = {'feat0': 0, 'feat1': 1, 'feat2': 2, 'feat3': 3}

    # Test with explicit list
    expert_config = {'important_features': ['feat0', 'feat1']}
    indices = _resolve_important_feature_indices(expert_config, feat2idx)
    assert indices == [0, 1]

    # Test with rules
    expert_config = {
        'rules': [
            {'feature': 'feat2', 'relation': '>=mean'},
            {'feature': 'feat3', 'important': True}
        ]
    }
    indices = _resolve_important_feature_indices(expert_config, feat2idx)
    assert 2 in indices and 3 in indices
    print(f"  ✓ Important feature resolution works. Indices: {indices}")


def test_adversarial_generator():
    """Test adversarial generator."""
    print("\nTesting adversarial generator...")

    gen = TabularGeneratorWithRealInput(
        input_dim=10,
        latent_dim=8,
        extreme_dim=1,
        hidden=64,
        depth=2
    )

    x = torch.randn(4, 10)
    z = torch.randn(4, 8)
    e = torch.randn(4, 1)

    x_adv = gen(x, z, e)
    assert x_adv.shape == x.shape
    print(f"  ✓ Adversarial generator works. Output shape: {x_adv.shape}")


def test_trainer_initialization():
    """Test MySATrainer initialization."""
    print("\nTesting MySATrainer initialization...")

    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 5),
        torch.nn.Sigmoid()
    )

    # Create trainer
    trainer = MySATrainer(
        model=model,
        lr=1e-3,
        lambda_smooth=0.01,
        lambda_expert=0.1,
        ig_steps=10,
        ig_batch_samples=16,
        ig_time_subsample=5,
        X_train_ref=torch.randn(100, 10)
    )

    assert trainer.device in ['cpu', 'cuda']
    assert trainer.lambda_smooth == 0.01
    assert trainer.lambda_expert == 0.1
    print(f"  ✓ MySATrainer initialization works.")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Advanced TEXGISA Functions")
    print("=" * 60)

    test_gpd_sampling()
    test_integrated_gradients()
    test_texgi_time_series()
    test_temporal_smoothness()
    test_expert_penalty()
    test_data_processing()
    test_resolve_important_features()
    test_adversarial_generator()
    test_trainer_initialization()

    print("\n" + "=" * 60)
    print("✅ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()