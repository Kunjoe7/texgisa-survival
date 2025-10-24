"""
Example demonstrating advanced TEXGISA features ported from SAWEB.

This example shows:
1. Using the MySATrainer for better training abstraction
2. Leveraging GPD sampling for adversarial baselines
3. Applying L2 norm-based expert penalties
4. Computing time-dependent integrated gradients
5. Using multimodal fusion (if data available)
"""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the advanced functions
import sys
sys.path.append('../src')

from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
from texgisa_survival.models.texgisa_advanced import (
    MySATrainer,
    TabularGeneratorWithRealInput,
    sample_extreme_code,
    train_adv_generator,
    integrated_gradients_time,
    texgi_time_series,
    attribution_temporal_l1,
    expert_penalty,
    _resolve_important_feature_indices,
    make_intervals,
    build_supervision,
    _standardize_fit,
)

# For multimodal support (optional)
from texgisa_survival.models.multimodal import (
    MultiModalFusionModel,
    ImageEncoder,
    SensorEncoder,
)


def generate_survival_data(n_samples=1000, n_features=20, random_state=42):
    """Generate synthetic survival data."""
    np.random.seed(random_state)

    # Generate features
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=random_state
    )

    # Generate survival times using Cox model simulation
    beta = np.random.randn(n_features) * 0.5
    beta[:5] = np.array([2, -1.5, 1, -0.5, 0.8])  # Strong effects for first 5 features

    # Linear predictor
    eta = X @ beta

    # Baseline hazard (Weibull)
    lambda_0 = 0.01
    rho = 1.5

    # Generate survival times
    U = np.random.uniform(0, 1, n_samples)
    T = (-np.log(U) / (lambda_0 * np.exp(eta))) ** (1/rho)

    # Add censoring
    C = np.random.exponential(scale=np.median(T) * 1.5, size=n_samples)
    time = np.minimum(T, C)
    event = (T <= C).astype(int)

    # Create DataFrame
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['duration'] = time
    df['event'] = event

    return df, feature_cols


def run_advanced_texgisa_example():
    """Main example demonstrating advanced features."""

    print("=== Advanced TEXGISA Example ===\n")

    # 1. Generate synthetic data
    print("1. Generating synthetic survival data...")
    df, feature_cols = generate_survival_data(n_samples=1000, n_features=20)
    print(f"   Generated {len(df)} samples with {len(feature_cols)} features")
    print(f"   Event rate: {df['event'].mean():.2%}\n")

    # 2. Prepare data with advanced functions
    print("2. Preparing data with advanced discretization...")
    df = make_intervals(df, duration_col='duration', event_col='event',
                       n_bins=30, method='quantile')
    num_bins = int(df['interval_number'].max()) + 1
    print(f"   Created {num_bins} time bins\n")

    # Split data
    X = df[feature_cols].values.astype(np.float32)
    durations = df['duration'].values
    events = df['event'].values
    intervals = df['interval_number'].values

    X_train, X_val, dur_train, dur_val, evt_train, evt_val, int_train, int_val = train_test_split(
        X, durations, events, intervals, test_size=0.2, random_state=42
    )

    # Create supervision
    y_train, m_train = build_supervision(int_train, evt_train, num_bins)
    y_val, m_val = build_supervision(int_val, evt_val, num_bins)

    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples\n")

    # 3. Define expert rules for important features
    print("3. Setting up expert rules...")
    expert_rules = {
        'important_features': ['feature_0', 'feature_1', 'feature_2'],  # Known important features
        'rules': [
            {'feature': 'feature_0', 'relation': '>=mean', 'weight': 2.0},
            {'feature': 'feature_1', 'relation': '>=mean', 'weight': 1.5},
        ]
    }
    feat2idx = {name: i for i, name in enumerate(feature_cols)}
    important_idx = _resolve_important_feature_indices(expert_rules, feat2idx)
    print(f"   Important features: {[feature_cols[i] for i in important_idx]}\n")

    # 4. Create model
    print("4. Creating enhanced model...")
    input_dim = X_train.shape[1]
    model = MultiTaskSurvivalNet(
        input_dim=input_dim,
        hidden_layers=[128, 64, 32],
        num_time_bins=num_bins,
        dropout=0.1,
        activation='relu'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}\n")

    # 5. Initialize advanced trainer
    print("5. Initializing MySATrainer with advanced features...")
    trainer = MySATrainer(
        model=model,
        lr=1e-3,
        device=device,
        lambda_smooth=0.01,  # Temporal smoothness on attributions
        lambda_expert=0.1,    # Expert penalty weight
        expert_rules=expert_rules,
        feat2idx=feat2idx,
        ig_steps=20,         # Integration steps for TEXGI
        latent_dim=16,       # Latent dimension for generator
        extreme_dim=1,       # Extreme code dimension
        ig_batch_samples=32,  # Subsample for efficiency
        ig_time_subsample=10, # Subsample time bins
        gen_epochs=100,      # Generator training epochs
        gen_batch=128,
        gen_lr=1e-3,
        gen_alpha_dist=1.0,
        X_train_ref=torch.tensor(X_train, dtype=torch.float32)
    )
    print("   Trainer configured with:")
    print(f"   - Lambda smooth: {trainer.lambda_smooth}")
    print(f"   - Lambda expert: {trainer.lambda_expert}")
    print(f"   - IG steps: {trainer.ig_steps}")
    print(f"   - Batch subsampling: {trainer.ig_batch_samples}")
    print(f"   - Time subsampling: {trainer.ig_time_subsample}\n")

    # 6. Train model with advanced features
    print("6. Training with advanced loss components...")

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    m_train_t = torch.tensor(m_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    m_val_t = torch.tensor(m_val, dtype=torch.float32)

    # Training loop
    epochs = 50
    batch_size = 64
    best_cindex = 0

    for epoch in range(epochs):
        # Mini-batch training
        n_samples = len(X_train_t)
        indices = torch.randperm(n_samples)

        epoch_losses = {'main': 0, 'smooth': 0, 'expert': 0}
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]

            # Get batch
            X_batch = X_train_t[batch_idx]
            y_batch = y_train_t[batch_idx]
            m_batch = m_train_t[batch_idx]

            # Training step
            loss_main, loss_smooth, loss_expert = trainer.step(
                X_batch, y_batch, m_batch
            )

            epoch_losses['main'] += loss_main
            epoch_losses['smooth'] += loss_smooth
            epoch_losses['expert'] += loss_expert
            n_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        # Evaluate
        cindex = trainer.evaluate_cindex(
            X_val_t, y_val_t, m_val_t,
            torch.tensor(dur_val), torch.tensor(evt_val)
        )

        if cindex > best_cindex:
            best_cindex = cindex

        if epoch % 10 == 0:
            print(f"   Epoch {epoch:3d} | Loss: {epoch_losses['main']:.4f} | "
                  f"Smooth: {epoch_losses['smooth']:.4f} | Expert: {epoch_losses['expert']:.4f} | "
                  f"C-index: {cindex:.4f}")

    print(f"\n   Best validation C-index: {best_cindex:.4f}\n")

    # 7. Compute TEXGI attributions
    print("7. Computing TEXGI attributions with adversarial baselines...")

    model.eval()
    with torch.no_grad():
        # Take a small sample for attribution
        n_attr_samples = 20
        X_attr = X_val_t[:n_attr_samples].to(device)

        # Compute TEXGI with adversarial baselines
        phi = texgi_time_series(
            model.to(device),
            X_attr,
            trainer.G,
            trainer.ref_stats,
            M=20,
            latent_dim=trainer.latent_dim,
            extreme_dim=trainer.extreme_dim,
            t_sample=5  # Sample 5 time points for efficiency
        )

        print(f"   Attribution shape: {phi.shape} [T, B, D]")

        # Aggregate importance
        imp_abs = phi.abs().mean(dim=(0, 1)).cpu().numpy()

        # Top features
        top_k = 5
        top_indices = np.argsort(imp_abs)[-top_k:][::-1]

        print(f"\n   Top {top_k} important features by TEXGI:")
        for idx in top_indices:
            print(f"   - {feature_cols[idx]}: {imp_abs[idx]:.4f}")

        # Check temporal smoothness
        smoothness = attribution_temporal_l1(phi).item()
        print(f"\n   Attribution temporal smoothness: {smoothness:.4f}")

        # Check expert penalty
        expert_pen = expert_penalty(phi, important_idx).item()
        print(f"   Expert penalty value: {expert_pen:.4f}\n")

    # 8. Demonstrate GPD sampling
    print("8. Demonstrating Generalized Pareto Distribution sampling...")
    extreme_codes = sample_extreme_code(
        batch_size=10,
        extreme_dim=1,
        device=device,
        xi=0.3,
        beta=1.0
    )
    print(f"   Extreme code samples (first 5): {extreme_codes[:5].squeeze().cpu().numpy()}")
    print(f"   Mean: {extreme_codes.mean().item():.4f}")
    print(f"   Std: {extreme_codes.std().item():.4f}\n")

    # 9. Show multimodal capability (structure only, no real data)
    print("9. Multimodal fusion model structure (demonstration)...")

    multimodal_config = {
        'tabular': {'input_dim': 20},
        'image': {'backbone': 'resnet18', 'pretrained': False},
        'sensor': {'input_channels': 3, 'backbone': 'cnn'}
    }

    multimodal_model = MultiModalFusionModel(
        modality_configs=multimodal_config,
        num_bins=num_bins,
        hidden=256,
        depth=2,
        dropout=0.2
    )

    total_params = sum(p.numel() for p in multimodal_model.parameters())
    print(f"   Created multimodal model with {total_params:,} parameters")
    print(f"   Modalities: {list(multimodal_config.keys())}")

    print("\n=== Example Complete ===")
    print("\nKey Advanced Features Demonstrated:")
    print("✓ MySATrainer with batch/time subsampling")
    print("✓ Generalized Pareto Distribution sampling")
    print("✓ Advanced adversarial generator training")
    print("✓ L2 norm-based expert penalties")
    print("✓ Time-dependent integrated gradients (TEXGI)")
    print("✓ Attribution temporal smoothness")
    print("✓ Multimodal fusion architecture")


if __name__ == "__main__":
    run_advanced_texgisa_example()