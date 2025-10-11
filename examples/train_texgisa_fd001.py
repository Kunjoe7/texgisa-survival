#!/usr/bin/env python3
# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Train TEXGISA model on FD001 turbine engine survival data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import sys
import os

# Add current directory to path to import texgisa_survival
sys.path.insert(0, os.getcwd())

# Import from local modules  
from models.texgisa import TEXGISAModel
from data import SurvivalDataset
from metrics import concordance_index, brier_score


def load_fd001_data():
    """Load FD001 turbine engine sensor data and survival labels."""
    print("Loading FD001 data...")
    
    # Load survival labels
    labels_df = pd.read_csv('FD001_censored_labels.csv')
    print(f"Loaded survival labels for {len(labels_df)} units")
    print(f"Event rate: {labels_df['event'].mean():.2%}")
    print(f"Median duration: {labels_df['duration'].median():.1f} cycles")
    
    # Load sensor data for each unit
    all_data = []
    fd001_dir = Path('FD001')
    
    for _, row in labels_df.iterrows():
        unit_file = row['file']
        duration = row['duration'] 
        event = row['event']
        
        # Load unit sensor data
        unit_path = fd001_dir / unit_file
        if unit_path.exists():
            unit_data = pd.read_csv(unit_path)
            
            # Extract features from the last few cycles (aggregated features)
            # Use statistics from last 10 cycles or all available cycles if less
            n_cycles = min(10, len(unit_data))
            recent_data = unit_data.tail(n_cycles)
            
            # Calculate statistical features for each sensor
            features = {}
            sensor_cols = [col for col in unit_data.columns if col.startswith('s') or col.startswith('setting')]
            
            for col in sensor_cols:
                features[f'{col}_mean'] = recent_data[col].mean()
                features[f'{col}_std'] = recent_data[col].std()
                features[f'{col}_min'] = recent_data[col].min() 
                features[f'{col}_max'] = recent_data[col].max()
                features[f'{col}_trend'] = recent_data[col].iloc[-1] - recent_data[col].iloc[0] if len(recent_data) > 1 else 0
            
            # Add survival info
            features['duration'] = duration
            features['event'] = event
            features['unit_id'] = int(unit_file.split('_')[2].split('.')[0])
            features['total_cycles'] = len(unit_data)
            
            all_data.append(features)
        else:
            print(f"Warning: {unit_file} not found")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    print(f"Created feature matrix with shape: {df.shape}")
    print(f"Features per sensor: mean, std, min, max, trend")
    
    return df


def train_texgisa_model(data_df):
    """Train TEXGISA model with expert rules."""
    print("\n" + "="*50)
    print("Training TEXGISA Model")
    print("="*50)
    
    # Create survival dataset
    dataset = SurvivalDataset(
        data=data_df,
        time_col='duration', 
        event_col='event',
        feature_cols=[col for col in data_df.columns if col not in ['duration', 'event', 'unit_id']]
    )
    
    # Preprocess data
    print("Preprocessing data...")
    dataset.handle_missing(strategy='median')
    dataset.normalize(method='standard')
    
    # Split data
    X_train, X_test, time_train, time_test, event_train, event_test = dataset.split(
        test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Initialize TEXGISA model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TEXGISAModel(device=device, random_state=42)
    
    print(f"Initialized TEXGISA model on {device}")
    
    # Train model
    print("\nTraining model...")
    model.fit(
        X_train.values, time_train.values, event_train.values,
        validation_data=(X_test.values, time_test.values, event_test.values),
        epochs=50,  # Reduced for faster training
        batch_size=16,
        learning_rate=0.001,
        early_stopping=True,
        patience=10,
        verbose=1
    )
    
    return model, (X_train, X_test, time_train, time_test, event_train, event_test)


def evaluate_model(model, data_splits):
    """Evaluate trained model and show results."""
    X_train, X_test, time_train, time_test, event_train, event_test = data_splits
    
    print("\n" + "="*50)
    print("Model Evaluation Results")  
    print("="*50)
    
    # Get predictions
    train_risk = model.predict_risk(X_train.values)
    test_risk = model.predict_risk(X_test.values)
    
    # Calculate metrics
    train_c_index = concordance_index(time_train.values, event_train.values, train_risk)
    test_c_index = concordance_index(time_test.values, event_test.values, test_risk)
    
    print("Training Performance:")
    print(f"  C-index: {train_c_index:.4f}")
    
    print("\nTest Performance:")
    print(f"  C-index: {test_c_index:.4f}")
    
    test_scores = {'c-index': test_c_index}
    
    # Get feature importance if available
    try:
        importance_dict = model.get_feature_importance(method='default')
        print("\n" + "="*30)
        print("Feature Importance")
        print("="*30)
        
        # Create DataFrame for visualization
        feature_names = [f'feature_{i}' for i in range(len(importance_dict['importance']))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_dict['importance']
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(15)
        
        for _, row in top_features.iterrows():
            print(f"{row['feature']:<25}: {row['importance']:.6f}")
            
    except Exception as e:
        print(f"Could not compute feature importance: {e}")
        importance_df = None
    
    # Plot feature importance
    if importance_df is not None:
        plt.figure(figsize=(12, 8))
        top_20 = importance_df.head(20)
        
        plt.barh(range(len(top_20)), top_20['importance'])
        plt.yticks(range(len(top_20)), top_20['feature'], fontsize=10)
        plt.xlabel('Importance Score')
        plt.title('Top 20 Feature Importance (TEXGISA Model)')
        plt.tight_layout()
        plt.gca().invert_yaxis()
        plt.savefig('texgisa_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved as 'texgisa_feature_importance.png'")
    
    # Predict survival probabilities for sample units
    print("\n" + "="*40)
    print("Sample Survival Predictions")
    print("="*40)
    
    try:
        # Get survival predictions for first 5 test samples
        sample_indices = list(range(min(5, len(X_test))))
        times = np.linspace(0, time_test.max(), 50)
        
        survival_probs = model.predict_survival(X_test.iloc[sample_indices].values, times)
        
        # Plot survival curves
        plt.figure(figsize=(12, 8))
        
        for i, idx in enumerate(sample_indices):
            plt.plot(times, survival_probs[i], 
                    label=f'Unit {idx+1} (Event={event_test.iloc[idx]}, Time={time_test.iloc[idx]:.0f})',
                    linewidth=2)
        
        plt.xlabel('Time (cycles)')
        plt.ylabel('Survival Probability')
        plt.title('Predicted Survival Curves (Sample Units)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('survival_curves_sample.png', dpi=300, bbox_inches='tight')
        print("Survival curves plot saved as 'survival_curves_sample.png'")
        
    except Exception as e:
        print(f"Could not generate survival curves: {e}")
    
    # Risk score distribution (already computed above)
    # train_risk = model.predict_risk(X_train.values)
    # test_risk = model.predict_risk(X_test.values)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(train_risk[event_train == 0], alpha=0.7, label='Censored', bins=20)
    plt.hist(train_risk[event_train == 1], alpha=0.7, label='Event', bins=20)
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.title('Training Set Risk Score Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(test_risk[event_test == 0], alpha=0.7, label='Censored', bins=20)
    plt.hist(test_risk[event_test == 1], alpha=0.7, label='Event', bins=20)
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')  
    plt.title('Test Set Risk Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('risk_score_distribution.png', dpi=300, bbox_inches='tight')
    print("Risk score distribution plot saved as 'risk_score_distribution.png'")
    
    return test_scores


def main():
    """Main training pipeline."""
    print("DHAI Survival Analysis - TEXGISA Training on FD001 Data")
    print("="*60)
    
    # Load data
    data_df = load_fd001_data()
    
    # Train model
    model, data_splits = train_texgisa_model(data_df)
    
    # Evaluate and show results
    final_scores = evaluate_model(model, data_splits)
    
    # Save model state
    print(f"\nSaving trained model...")
    try:
        import torch
        model_state = model.get_state()
        torch.save(model_state, 'texgisa_fd001_model.pkl')
        print("Model saved as 'texgisa_fd001_model.pkl'")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("Generated files:")
    print("- texgisa_fd001_model.pkl (trained model)")
    print("- texgisa_feature_importance.png")
    print("- survival_curves_sample.png") 
    print("- risk_score_distribution.png")
    
    return model, final_scores


if __name__ == "__main__":
    model, scores = main()