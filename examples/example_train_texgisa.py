#!/usr/bin/env python3
# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Example: Train TEXGISA model on FD001 turbine engine survival data using DHAI Survival package.

This script demonstrates the clean and easy way to use the texgisa_survival package
for survival analysis with the TEXGISA model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Clean import of the texgisa_survival package
from texgisa_survival import SurvivalModel, SurvivalDataset, quick_analysis


def load_fd001_data():
    """Load and prepare FD001 turbine engine data for survival analysis."""
    print("Loading FD001 turbine engine data...")
    
    # Load survival labels
    labels_df = pd.read_csv('FD001_censored_labels.csv')
    print(f"Found {len(labels_df)} turbine units")
    print(f"Event rate: {labels_df['event'].mean():.2%}")
    
    # Load and aggregate sensor data for each unit
    all_data = []
    fd001_dir = Path('FD001')
    
    for _, row in labels_df.iterrows():
        unit_file = row['file']
        unit_path = fd001_dir / unit_file
        
        if unit_path.exists():
            unit_data = pd.read_csv(unit_path)
            
            # Extract aggregated features from last 10 cycles
            n_cycles = min(10, len(unit_data))
            recent_data = unit_data.tail(n_cycles)
            
            # Calculate statistical features for each sensor
            features = {}
            sensor_cols = [col for col in unit_data.columns 
                          if col.startswith('s') or col.startswith('setting')]
            
            for col in sensor_cols:
                features[f'{col}_mean'] = recent_data[col].mean()
                features[f'{col}_std'] = recent_data[col].std()
                features[f'{col}_max'] = recent_data[col].max()
                features[f'{col}_trend'] = (recent_data[col].iloc[-1] - recent_data[col].iloc[0] 
                                           if len(recent_data) > 1 else 0)
            
            # Add survival information
            features['duration'] = row['duration']
            features['event'] = row['event']
            features['unit_id'] = int(unit_file.split('_')[2].split('.')[0])
            
            all_data.append(features)
    
    df = pd.DataFrame(all_data)
    print(f"Created dataset with {df.shape[0]} samples and {df.shape[1]} features")
    
    return df


def main():
    """Main demonstration of clean texgisa_survival package usage."""
    
    print("="*60)
    print("DHAI Survival Package - TEXGISA Model Training Example")
    print("="*60)
    
    # Step 1: Load data
    data_df = load_fd001_data()
    
    # Step 2: Create SurvivalDataset (using package's data preprocessing)
    print("\nCreating SurvivalDataset...")
    dataset = SurvivalDataset(
        data=data_df,
        time_col='duration',
        event_col='event'
    )
    
    # Apply preprocessing
    dataset.handle_missing(strategy='median')
    dataset.normalize(method='standard')
    
    # Get dataset summary
    summary = dataset.get_summary()
    print("\nDataset Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Split data
    X_train, X_test, time_train, time_test, event_train, event_test = dataset.split(
        test_size=0.2,
        random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Step 3: Initialize TEXGISA model using clean API
    print("\n" + "="*40)
    print("Training TEXGISA Model")
    print("="*40)
    
    model = SurvivalModel(
        model_type='texgisa',
        random_state=42,
        verbose=1
    )
    
    # Step 4: Add expert rules (domain knowledge)
    print("\nAdding expert rules based on domain knowledge...")
    
    # Temperature sensors - higher values indicate stress
    model.add_expert_rule('s4_mean', '>', 'mean', sign='+1', weight=1.5)
    model.add_expert_rule('s11_mean', '>', 'mean', sign='+1', weight=1.2)
    
    # Vibration sensors - higher variation indicates degradation
    model.add_expert_rule('s6_std', '>', 'mean', sign='+1', weight=1.3)
    model.add_expert_rule('s7_std', '>', 'mean', sign='+1', weight=1.1)
    
    # Trend indicators - increasing trends suggest deterioration
    model.add_expert_rule('s4_trend', '>', 0, sign='+1', weight=1.0)
    model.add_expert_rule('s11_trend', '>', 0, sign='+1', weight=1.0)
    
    # Step 5: Train the model
    print("\nTraining model...")
    model.fit(
        X_train, time_train, event_train,
        validation_data=(X_test, time_test, event_test),
        epochs=50,
        batch_size=16,
        learning_rate=0.001,
        early_stopping=True,
        patience=10,
        lambda_expert=0.1  # Weight for expert knowledge
    )
    
    # Step 6: Evaluate model performance
    print("\n" + "="*40)
    print("Model Evaluation")
    print("="*40)
    
    # Evaluate on training set
    train_scores = model.evaluate(
        X_train, time_train, event_train,
        metrics=['c-index', 'brier_score']
    )
    print("\nTraining Performance:")
    for metric, score in train_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Evaluate on test set
    test_scores = model.evaluate(
        X_test, time_test, event_test,
        metrics=['c-index', 'brier_score']
    )
    print("\nTest Performance:")
    for metric, score in test_scores.items():
        print(f"  {metric}: {score:.4f}")
    
    # Step 7: Get feature importance (TEXGISA special feature)
    print("\n" + "="*40)
    print("Feature Importance (TEXGI Method)")
    print("="*40)
    
    importance_df = model.get_feature_importance(method='texgi', n_steps=50)
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string())
    
    # Step 8: Generate predictions
    print("\n" + "="*40)
    print("Generating Predictions")
    print("="*40)
    
    # Predict risk scores
    test_risk = model.predict_risk(X_test)
    print(f"\nRisk scores - Mean: {test_risk.mean():.4f}, Std: {test_risk.std():.4f}")
    
    # Predict survival curves for sample units
    sample_indices = [0, 1, 2]  # First 3 test samples
    times = np.linspace(0, time_test.max(), 50)
    survival_probs = model.predict_survival(
        X_test.iloc[sample_indices], 
        times=times
    )
    
    # Step 9: Visualize results
    print("\n" + "="*40)
    print("Creating Visualizations")
    print("="*40)
    
    # Plot survival curves
    model.plot_survival_curves(
        X_test.iloc[sample_indices],
        times=times,
        labels=[f"Unit {i+1}" for i in sample_indices],
        title="TEXGISA Predicted Survival Curves - Sample Units",
        save_path="texgisa_survival_curves.png"
    )
    print("Saved survival curves plot")
    
    # Plot risk score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(test_risk[event_test == 0], alpha=0.6, label='Censored', bins=15)
    plt.hist(test_risk[event_test == 1], alpha=0.6, label='Event', bins=15)
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.title('TEXGISA Risk Score Distribution - Test Set')
    plt.legend()
    plt.savefig('texgisa_risk_distribution.png')
    plt.close()
    print("Saved risk distribution plot")
    
    # Step 10: Save the trained model
    print("\n" + "="*40)
    print("Saving Model")
    print("="*40)
    
    model.save('texgisa_fd001_trained.pkl')
    print("Model saved as 'texgisa_fd001_trained.pkl'")
    
    # Demonstrate loading
    print("\nDemonstrating model loading...")
    loaded_model = SurvivalModel()
    loaded_model.load('texgisa_fd001_trained.pkl')
    print("Model successfully loaded!")
    
    # Verify loaded model works
    loaded_risk = loaded_model.predict_risk(X_test.iloc[:5])
    print(f"Loaded model predictions (first 5): {loaded_risk}")
    
    print("\n" + "="*60)
    print("Example Complete!")
    print("="*60)
    print("\nThis example demonstrated:")
    print("1. Clean import of texgisa_survival package")
    print("2. Data preprocessing with SurvivalDataset")
    print("3. Model initialization with unified API")
    print("4. Adding expert rules for TEXGISA")
    print("5. Model training and evaluation")
    print("6. Feature importance analysis")
    print("7. Generating predictions and visualizations")
    print("8. Saving and loading trained models")
    
    return model, test_scores


def demo_quick_analysis():
    """Demonstrate the quick_analysis convenience function."""
    
    print("\n" + "="*60)
    print("Bonus: Quick Analysis Demo")
    print("="*60)
    
    # Load data
    data_df = load_fd001_data()
    
    # Run quick analysis - compares multiple models automatically
    print("\nRunning quick analysis with multiple models...")
    results = quick_analysis(
        data=data_df,
        time_col='duration',
        event_col='event',
        model_types=['deepsurv', 'texgisa'],
        test_size=0.2
    )
    
    print("\nQuick Analysis Results:")
    print("Model Comparison:")
    print(results['comparison'])
    
    print("\nFeature importance available for each model in results['feature_importance']")
    print("Trained models available in results['models']")
    
    return results


if __name__ == "__main__":
    # Run main example
    model, scores = main()
    
    # Optional: Run quick analysis demo
    # Uncomment to see quick_analysis in action
    # results = demo_quick_analysis()