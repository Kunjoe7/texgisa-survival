#!/usr/bin/env python3
# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Train TEXGISA model on PCAM survival data with expert rules for medical imaging.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.insert(0, os.getcwd())

from models.texgisa import TEXGISAModel
from metrics import concordance_index


def extract_pcam_features():
    """Extract features from PCAM medical images."""
    print("Loading PCAM Survival Data for TEXGISA")
    print("="*60)
    
    # Load labels
    labels_df = pd.read_csv('pcam_survival_5k_labels.csv')
    print(f"Total images: {len(labels_df)}")
    print(f"Event rate: {labels_df['event'].mean():.2%}")
    print(f"Median survival: {labels_df['duration'].median():.1f} days")
    
    # Use a reasonable subset for TEXGISA training
    subset_size = 500
    subset_df = labels_df.head(subset_size).copy().reset_index(drop=True)
    
    print(f"\nUsing {subset_size} images for TEXGISA training")
    print(f"Subset event rate: {subset_df['event'].mean():.2%}")
    print(f"Class distribution: Normal={sum(subset_df['class_label']==0)}, Pathological={sum(subset_df['class_label']==1)}")
    
    # Extract image features
    image_dir = Path('pcam_images')
    features_list = []
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Slightly larger for better feature extraction
        transforms.ToTensor()
    ])
    
    print("\nExtracting medical image features...")
    
    for idx, row in subset_df.iterrows():
        img_path = image_dir / row['image']
        
        if img_path.exists():
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img)
                
                # Extract comprehensive features for medical imaging
                features = {}
                
                # RGB channel statistics (important for pathology)
                for c, color in enumerate(['R', 'G', 'B']):
                    channel = img_tensor[c].numpy().flatten()
                    features[f'{color}_mean'] = channel.mean()
                    features[f'{color}_std'] = channel.std()
                    features[f'{color}_skew'] = np.mean((channel - channel.mean())**3) / (channel.std()**3 + 1e-7)
                    features[f'{color}_kurt'] = np.mean((channel - channel.mean())**4) / (channel.std()**4 + 1e-7)
                    features[f'{color}_p10'] = np.percentile(channel, 10)
                    features[f'{color}_p90'] = np.percentile(channel, 90)
                
                # Grayscale features
                gray = img_tensor.mean(dim=0).numpy()
                features['brightness'] = gray.mean()
                features['contrast'] = gray.std()
                features['edge_density'] = np.std(np.gradient(gray))
                
                # Texture features (important for tissue analysis)
                features['entropy'] = -np.sum(np.histogram(gray.flatten(), bins=32)[0] * 
                                            np.log(np.histogram(gray.flatten(), bins=32)[0] + 1e-10))
                
                # Color ratios (important for staining patterns)
                r_mean = img_tensor[0].mean().item()
                g_mean = img_tensor[1].mean().item()
                b_mean = img_tensor[2].mean().item()
                
                features['RG_ratio'] = r_mean / (g_mean + 1e-7)
                features['RB_ratio'] = r_mean / (b_mean + 1e-7)
                features['GB_ratio'] = g_mean / (b_mean + 1e-7)
                
                # Add metadata
                features['class_label'] = row['class_label']
                
                features_list.append(features)
                
                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(subset_df)} images...")
                    
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
                # Add default features
                features = {f: 0.5 for f in ['R_mean', 'R_std', 'R_skew', 'R_kurt', 'R_p10', 'R_p90',
                                            'G_mean', 'G_std', 'G_skew', 'G_kurt', 'G_p10', 'G_p90',
                                            'B_mean', 'B_std', 'B_skew', 'B_kurt', 'B_p10', 'B_p90',
                                            'brightness', 'contrast', 'edge_density', 'entropy',
                                            'RG_ratio', 'RB_ratio', 'GB_ratio']}
                features['class_label'] = row['class_label']
                features_list.append(features)
    
    # Create feature matrix
    feature_df = pd.DataFrame(features_list)
    
    # Add survival data
    feature_df['duration'] = subset_df['duration'].values[:len(feature_df)]
    feature_df['event'] = subset_df['event'].values[:len(feature_df)]
    
    print(f"\nFeature extraction complete: {feature_df.shape}")
    feature_cols = [col for col in feature_df.columns if col not in ['duration', 'event', 'class_label']]
    print(f"Features extracted: {len(feature_cols)}")
    
    return feature_df


def train_texgisa_model(data_df):
    """Train TEXGISA model with medical imaging expert rules."""
    print("\n" + "="*60)
    print("Training TEXGISA Model with Medical Expert Rules")
    print("="*60)
    
    # Prepare features
    feature_cols = [col for col in data_df.columns 
                   if col not in ['duration', 'event', 'class_label']]
    
    X = data_df[feature_cols].values.astype(np.float32)
    time = data_df['duration'].values.astype(np.float32)
    event = data_df['event'].values.astype(np.int32)
    classes = data_df['class_label'].values
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, time_train, time_test, event_train, event_test, class_train, class_test = train_test_split(
        X, time, event, classes, test_size=0.25, random_state=42, stratify=event
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train event rate: {event_train.mean():.2%}")
    print(f"Test event rate: {event_test.mean():.2%}")
    
    # Initialize TEXGISA model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TEXGISAModel(
        device=device,
        random_state=42,
        num_time_bins=20,  # Fewer bins for smaller dataset
        hidden_layers=[64, 32, 16],  # Smaller network
        dropout=0.3
    )
    
    print(f"\nTEXGISA model initialized on {device}")
    
    # Define expert rules for medical imaging
    print("\nApplying medical imaging expert rules...")
    
    # Create expert knowledge matrix
    n_features = X_train.shape[1]
    expert_knowledge = np.zeros((n_features,))
    
    # Map feature names to indices
    feature_to_idx = {name: i for i, name in enumerate(feature_cols)}
    
    # Expert Rule 1: Red channel features (important for H&E staining)
    if 'R_mean' in feature_to_idx:
        expert_knowledge[feature_to_idx['R_mean']] = 0.8  # Higher red often indicates different tissue
    if 'R_std' in feature_to_idx:
        expert_knowledge[feature_to_idx['R_std']] = 0.6  # Variation in red is important
    
    # Expert Rule 2: Texture features (critical for pathology)
    if 'entropy' in feature_to_idx:
        expert_knowledge[feature_to_idx['entropy']] = 1.0  # Texture complexity is very important
    if 'edge_density' in feature_to_idx:
        expert_knowledge[feature_to_idx['edge_density']] = 0.7  # Edge patterns matter
    
    # Expert Rule 3: Color ratios (staining patterns)
    if 'RG_ratio' in feature_to_idx:
        expert_knowledge[feature_to_idx['RG_ratio']] = 0.5
    if 'RB_ratio' in feature_to_idx:
        expert_knowledge[feature_to_idx['RB_ratio']] = 0.5
    
    # Expert Rule 4: Contrast and brightness
    if 'contrast' in feature_to_idx:
        expert_knowledge[feature_to_idx['contrast']] = 0.6
    if 'brightness' in feature_to_idx:
        expert_knowledge[feature_to_idx['brightness']] = 0.4
    
    print(f"Applied {np.sum(expert_knowledge > 0)} expert rules")
    
    # Train model
    print("\nTraining TEXGISA model...")
    model.fit(
        X_train, time_train, event_train,
        validation_data=(X_test, time_test, event_test),
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        early_stopping=True,
        patience=15,
        verbose=1,
        expert_knowledge=expert_knowledge,
        lambda_expert=0.1  # Weight for expert knowledge
    )
    
    return model, (X_train, X_test, time_train, time_test, event_train, event_test, class_train, class_test), scaler, feature_cols


def evaluate_texgisa_model(model, data_splits, feature_cols):
    """Evaluate TEXGISA model performance."""
    X_train, X_test, time_train, time_test, event_train, event_test, class_train, class_test = data_splits
    
    print("\n" + "="*60)
    print("TEXGISA Model Evaluation Results")
    print("="*60)
    
    # Get predictions
    train_risk = model.predict_risk(X_train)
    test_risk = model.predict_risk(X_test)
    
    # Calculate C-index
    train_c_index = concordance_index(time_train, event_train, train_risk)
    test_c_index = concordance_index(time_test, event_test, test_risk)
    
    print(f"Training C-index: {train_c_index:.4f}")
    print(f"Test C-index: {test_c_index:.4f}")
    
    # Performance by class
    print("\nPerformance by pathological class:")
    for class_val in [0, 1]:
        class_name = "Normal" if class_val == 0 else "Pathological"
        mask = class_test == class_val
        
        if mask.sum() > 0:
            class_risk = test_risk[mask]
            class_time = time_test[mask]
            class_event = event_test[mask]
            
            class_c_index = concordance_index(class_time, class_event, class_risk)
            print(f"  {class_name} (n={mask.sum()}): C-index = {class_c_index:.4f}")
            print(f"    Mean risk: {class_risk.mean():.4f} ± {class_risk.std():.4f}")
    
    # Feature importance using TEXGI
    print("\nFeature Importance (TEXGI method):")
    try:
        importance_dict = model.get_feature_importance(method='texgi', n_steps=30)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance_dict['importance'],
            'std': importance_dict.get('std', np.zeros(len(feature_cols)))
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']:<15}: {row['importance']:.6f} ± {row['std']:.6f}")
            
    except Exception as e:
        print(f"Could not compute TEXGI importance: {e}")
        importance_df = None
    
    # Risk analysis
    print("\nRisk Score Analysis:")
    print(f"Overall test risk - Mean: {test_risk.mean():.4f}, Std: {test_risk.std():.4f}")
    
    event_risks = test_risk[event_test == 1]
    censored_risks = test_risk[event_test == 0]
    print(f"Event cases - Mean risk: {event_risks.mean():.4f}, N: {len(event_risks)}")
    print(f"Censored cases - Mean risk: {censored_risks.mean():.4f}, N: {len(censored_risks)}")
    
    # Create visualizations
    create_texgisa_visualizations(model, data_splits, feature_cols, importance_df)
    
    return {
        'train_c_index': train_c_index,
        'test_c_index': test_c_index,
        'importance_df': importance_df
    }


def create_texgisa_visualizations(model, data_splits, feature_cols, importance_df):
    """Create comprehensive visualizations for TEXGISA results."""
    X_train, X_test, time_train, time_test, event_train, event_test, class_train, class_test = data_splits
    
    # Get predictions
    test_risk = model.predict_risk(X_test)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Risk distribution by event status
    ax1 = plt.subplot(2, 4, 1)
    ax1.hist(test_risk[event_test == 0], alpha=0.6, label='Censored', bins=20, color='blue')
    ax1.hist(test_risk[event_test == 1], alpha=0.6, label='Event', bins=20, color='red')
    ax1.set_xlabel('Risk Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Risk Distribution by Event Status')
    ax1.legend()
    
    # 2. Risk by pathological class
    ax2 = plt.subplot(2, 4, 2)
    normal_risks = test_risk[class_test == 0]
    path_risks = test_risk[class_test == 1]
    
    ax2.violinplot([normal_risks, path_risks], positions=[0, 1], showmeans=True)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Normal', 'Pathological'])
    ax2.set_ylabel('Risk Score')
    ax2.set_title('Risk Distribution by Class')
    
    # 3. Survival time vs risk
    ax3 = plt.subplot(2, 4, 3)
    colors = ['blue' if c == 0 else 'red' for c in class_test]
    ax3.scatter(test_risk, time_test, c=colors, alpha=0.6)
    ax3.set_xlabel('Risk Score')
    ax3.set_ylabel('Survival Time')
    ax3.set_title('Risk vs Survival Time\n(Blue=Normal, Red=Path)')
    
    # 4. Feature importance (if available)
    ax4 = plt.subplot(2, 4, 4)
    if importance_df is not None:
        top_features = importance_df.head(10)
        ax4.barh(range(len(top_features)), top_features['importance'])
        ax4.set_yticks(range(len(top_features)))
        ax4.set_yticklabels(top_features['feature'], fontsize=8)
        ax4.set_xlabel('TEXGI Importance')
        ax4.set_title('Top 10 Feature Importance')
    
    # 5. Survival curves by risk groups
    ax5 = plt.subplot(2, 4, 5)
    try:
        times = np.linspace(0, time_test.max(), 50)
        
        # Get survival probabilities for high and low risk groups
        risk_median = np.median(test_risk)
        high_risk_idx = np.where(test_risk >= risk_median)[0][:3]  # Top 3 high risk
        low_risk_idx = np.where(test_risk < risk_median)[0][:3]   # Top 3 low risk
        
        if len(high_risk_idx) > 0:
            surv_probs_high = model.predict_survival(X_test[high_risk_idx], times)
            for i in range(len(high_risk_idx)):
                ax5.plot(times, surv_probs_high[i], 'r-', alpha=0.7, linewidth=1.5)
        
        if len(low_risk_idx) > 0:
            surv_probs_low = model.predict_survival(X_test[low_risk_idx], times)
            for i in range(len(low_risk_idx)):
                ax5.plot(times, surv_probs_low[i], 'b-', alpha=0.7, linewidth=1.5)
        
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Survival Probability')
        ax5.set_title('Survival Curves\n(Red=High Risk, Blue=Low Risk)')
        ax5.grid(True, alpha=0.3)
        
    except Exception as e:
        ax5.text(0.5, 0.5, f'Could not generate\nsurvival curves:\n{e}', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Model performance comparison
    ax6 = plt.subplot(2, 4, 6)
    train_c_index = concordance_index(time_train, event_train, model.predict_risk(X_train))
    test_c_index = concordance_index(time_test, event_test, test_risk)
    
    bars = ax6.bar(['Training', 'Test'], [train_c_index, test_c_index], 
                   color=['lightgreen', 'lightcoral'])
    ax6.set_ylabel('C-index')
    ax6.set_title('TEXGISA Performance')
    ax6.set_ylim(0, 1)
    
    for bar, val in zip(bars, [train_c_index, test_c_index]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Event rate by class
    ax7 = plt.subplot(2, 4, 7)
    event_rates = []
    for class_val in [0, 1]:
        mask = class_test == class_val
        if mask.sum() > 0:
            event_rate = event_test[mask].mean()
            event_rates.append(event_rate)
        else:
            event_rates.append(0)
    
    bars = ax7.bar(['Normal', 'Pathological'], event_rates, 
                   color=['lightblue', 'lightcoral'])
    ax7.set_ylabel('Event Rate')
    ax7.set_title('Event Rate by Class')
    ax7.set_ylim(0, 1)
    
    for bar, rate in zip(bars, event_rates):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')
    
    # 8. Risk score heatmap by features
    ax8 = plt.subplot(2, 4, 8)
    if importance_df is not None:
        # Show correlation between top features and risk
        top_feature_names = importance_df.head(5)['feature'].tolist()
        top_feature_idx = [feature_cols.index(f) for f in top_feature_names if f in feature_cols]
        
        if len(top_feature_idx) > 0:
            correlations = []
            for idx in top_feature_idx:
                corr = np.corrcoef(X_test[:, idx], test_risk)[0, 1]
                correlations.append(corr)
            
            ax8.bar(range(len(correlations)), correlations)
            ax8.set_xticks(range(len(correlations)))
            ax8.set_xticklabels([feature_cols[idx] for idx in top_feature_idx], 
                               rotation=45, ha='right', fontsize=8)
            ax8.set_ylabel('Correlation with Risk')
            ax8.set_title('Feature-Risk Correlation')
    
    plt.suptitle('TEXGISA Model Results on PCAM Medical Images', fontsize=16)
    plt.tight_layout()
    plt.savefig('pcam_texgisa_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization saved as 'pcam_texgisa_results.png'")


def main():
    """Main execution pipeline."""
    print("PCAM Survival Analysis with TEXGISA Model")
    print("="*60)
    
    try:
        # Extract features
        data_df = extract_pcam_features()
        
        # Train TEXGISA model
        model, data_splits, scaler, feature_cols = train_texgisa_model(data_df)
        
        # Evaluate model
        results = evaluate_texgisa_model(model, data_splits, feature_cols)
        
        # Save model
        print(f"\nSaving TEXGISA model...")
        model_state = model.get_state()
        torch.save({
            'model_state': model_state,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'results': results
        }, 'pcam_texgisa_model.pkl')
        
        print("\n" + "="*60)
        print("TEXGISA Analysis Complete!")
        print("="*60)
        print(f"Final Performance:")
        print(f"  Training C-index: {results['train_c_index']:.4f}")
        print(f"  Test C-index: {results['test_c_index']:.4f}")
        print("\nFiles generated:")
        print("  - pcam_texgisa_model.pkl (trained model)")
        print("  - pcam_texgisa_results.png (comprehensive results)")
        
        return model, results
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    model, results = main()