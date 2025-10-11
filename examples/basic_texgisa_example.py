#!/usr/bin/env python3
# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Basic example: Train TexGISa model on synthetic survival data.

This script demonstrates the basic usage of the texgisa_survival package
with the TexGISa model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import from texgisa_survival
from texgisa_survival import TexGISa


def generate_synthetic_data(n_samples=1000, n_features=10, random_state=42):
    """Generate synthetic survival data for demonstration."""
    np.random.seed(random_state)

    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Generate survival times (exponential distribution)
    # Higher values of feature 0 and 1 increase risk
    baseline_hazard = 0.01
    risk = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
    time = np.random.exponential(1 / (baseline_hazard * risk))

    # Generate censoring (30% censoring rate)
    censoring_time = np.random.exponential(1.5 * time.mean(), n_samples)
    event = (time <= censoring_time).astype(np.float32)
    time = np.minimum(time, censoring_time).astype(np.float32)

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)

    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Event rate: {event.mean():.1%}")
    print(f"Mean survival time: {time.mean():.2f}")

    return X_df, time, event


def main():
    """Main demonstration of TexGISa model."""

    print("="*60)
    print("TexGISa Basic Example")
    print("="*60)

    # Step 1: Generate synthetic data
    print("\n1. Generating synthetic survival data...")
    X, time, event = generate_synthetic_data(n_samples=500, n_features=10)

    # Step 2: Split data
    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
        X, time, event, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Step 3: Initialize TexGISa model
    print("\n3. Initializing TexGISa model...")
    model = TexGISa(
        hidden_layers=[64, 32],
        num_time_bins=10,
        dropout=0.1,
        lambda_expert=0.1,
        ig_steps=20,
        random_state=42,
        verbose=1
    )

    # Step 4: Add expert knowledge rules
    print("\n4. Adding expert knowledge rules...")
    # We know feature_0 and feature_1 increase risk (from data generation)
    model.add_expert_rule('feature_0', '>', 'mean', sign='+1', weight=1.5)
    model.add_expert_rule('feature_1', '>', 'mean', sign='+1', weight=1.2)

    # Step 5: Train the model
    print("\n5. Training TexGISa model...")
    model.fit(
        X_train, time_train, event_train,
        validation_data=(X_test, time_test, event_test),
        epochs=30,
        batch_size=32,
        learning_rate=0.001,
        early_stopping=True,
        patience=10
    )

    # Step 6: Evaluate model
    print("\n6. Evaluating model performance...")
    test_scores = model.evaluate(
        X_test, time_test, event_test,
        metrics=['c-index']
    )
    print(f"\nTest C-index: {test_scores['c-index']:.4f}")

    # Step 7: Get feature importance
    print("\n7. Computing TEXGI feature importance...")
    importance_df = model.get_feature_importance(method='texgi')
    print("\nTop 5 Most Important Features:")
    print(importance_df.head(5).to_string(index=False))

    # Step 8: Make predictions
    print("\n8. Making predictions...")
    risk_scores = model.predict_risk(X_test)
    survival_probs = model.predict_survival(X_test)

    print(f"Risk scores - Mean: {risk_scores.mean():.4f}, Std: {risk_scores.std():.4f}")
    print(f"Survival probabilities shape: {survival_probs.shape}")

    # Step 9: Visualize results
    print("\n9. Creating visualizations...")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(10)
    plt.barh(top_features['feature'], top_features['importance'])
    plt.xlabel('TEXGI Importance')
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig('texgisa_feature_importance.png')
    plt.close()
    print("Saved feature importance plot to 'texgisa_feature_importance.png'")

    # Plot survival curves for sample individuals
    sample_indices = [0, 10, 20]
    times = np.linspace(0, time_test.max(), 50)
    survival_curves = model.predict_survival(X_test.iloc[sample_indices], times=times)

    plt.figure(figsize=(10, 6))
    for i, idx in enumerate(sample_indices):
        plt.plot(times, survival_curves[i], label=f'Individual {idx}', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.title('TexGISa Predicted Survival Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('texgisa_survival_curves.png')
    plt.close()
    print("Saved survival curves plot to 'texgisa_survival_curves.png'")

    # Step 10: Save the model
    print("\n10. Saving trained model...")
    model.save('texgisa_trained_model.pkl')
    print("Model saved to 'texgisa_trained_model.pkl'")

    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    print("\nThis example demonstrated:")
    print("✓ Data generation and splitting")
    print("✓ Model initialization")
    print("✓ Adding expert knowledge rules")
    print("✓ Model training with validation")
    print("✓ Performance evaluation")
    print("✓ TEXGI feature importance")
    print("✓ Survival prediction")
    print("✓ Visualization")
    print("✓ Model saving")

    return model


if __name__ == "__main__":
    model = main()
