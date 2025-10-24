# DHAI Survival Analysis: TexGISa

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Time-dependent EXtreme Gradient Integration for Survival Analysis (TexGISa)** - An interpretable deep learning framework for survival analysis with expert knowledge integration.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Capabilities](#core-capabilities)
- [Usage Examples](#usage-examples)
- [Advanced Features](#advanced-features)
  - [MySATrainer](#mysatrainer-enhanced-training-infrastructure)
  - [Generalized Pareto Distribution Sampling](#generalized-pareto-distribution-sampling)
  - [Time-Dependent Integrated Gradients](#time-dependent-integrated-gradients)
  - [L2 Norm-Based Expert Penalties](#l2-norm-based-expert-penalties)
  - [Multimodal Architecture](#multimodal-architecture)
- [API Documentation](#api-documentation)
- [Performance](#performance)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

TexGISa is a novel deep learning framework for survival analysis that combines the predictive power of neural networks with interpretability through **Time-dependent EXtreme Gradient Integration (TEXGI)**. Unlike traditional black-box deep learning models, TexGISa provides:

- **Time-dependent feature importance** via Expected Gradients
- **Expert knowledge integration** during model training
- **Interpretable predictions** with attribution explanations
- **Discrete-time hazard modeling** for flexible survival prediction

This implementation is based on our research published at **ICDM 2024**: *"TexGISa: Interpretable and Interactive Deep Survival Analysis with Time-dependent Extreme Gradient Integration"*.

### What Makes TexGISa Different?

**Traditional Survival Models:**
- Cox PH: Linear assumptions, limited flexibility
- Random Survival Forests: Black-box, no feature importance over time
- Standard Neural Networks: Lack interpretability

**TexGISa:**
- ✅ Deep learning flexibility without black-box limitations
- ✅ Time-dependent feature importance using TEXGI
- ✅ Expert knowledge constraints during training
- ✅ Full survival curve prediction with explanations

## Key Features

### 🧠 Interpretable Deep Learning
- **TEXGI (Time-dependent EXtreme Gradient Integration)**: Compute feature importance using Expected Gradients
- **Time-dependent attributions**: Understand how feature importance changes over time
- **Per-time-bin integrated gradients**: Advanced attribution computation for each time interval
- **Attribution visualization**: Built-in tools for explaining predictions

### 👨‍⚕️ Expert Knowledge Integration
- **Expert rules**: Incorporate domain knowledge as constraints
- **L2 norm-based expert penalties**: Advanced constraint formulation following Equation 18
- **Rule-based training**: Penalize model when attributions contradict expert knowledge
- **Flexible rule specification**: Support for feature conditions and expected effects

### 🎯 Flexible Survival Modeling
- **Discrete-time hazards**: Model hazard at each time interval
- **Multi-task learning**: Joint prediction across time bins
- **Temporal smoothness**: Optional regularization for smooth hazard curves
- **Advanced adversarial baselines**: Generalized Pareto Distribution sampling for extreme baselines

### 🌐 Multimodal Support (Advanced)
- **Image data**: ResNet and Vision Transformer encoders
- **Sensor/time-series**: CNN and Transformer encoders
- **Tabular data**: MLP encoders
- **Gating-based fusion**: Learnable attention for modality combination
- **Asset streaming**: Efficient on-disk data loading

### ⚡ Production Ready
- **GPU acceleration**: Full PyTorch implementation
- **Scikit-learn compatible**: Familiar `.fit()` and `.predict()` API
- **MySATrainer**: Advanced training infrastructure with batch/time subsampling
- **Type hints**: Complete type annotations throughout
- **Comprehensive testing**: Extensive test suite

## Installation

### From PyPI (Recommended)

```bash
pip install texgisa-survival
```

### From Source

```bash
git clone https://github.com/dhai-lab/texgisa_survival.git
cd texgisa_survival
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/dhai-lab/texgisa_survival.git
cd texgisa_survival
pip install -e ".[dev]"
```

### Requirements

- Python 3.8 or higher
- PyTorch 1.9.0 or higher
- NumPy, Pandas, scikit-learn
- See `pyproject.toml` for complete list

## Quick Start

```python
import numpy as np
from texgisa_survival import TexGISa

# Prepare your data
X = np.random.randn(1000, 10)  # Features
time = np.random.exponential(100, 1000)  # Time to event
event = np.random.binomial(1, 0.7, 1000)  # Event indicator (1=event, 0=censored)

# Create and train model
model = TexGISa(
    hidden_layers=[64, 32],
    num_time_bins=10,
    dropout=0.1,
    lambda_expert=0.1,  # Weight for expert knowledge
    ig_steps=20,  # Integration steps for TEXGI
    random_state=42
)

# Optional: Add expert knowledge
model.add_expert_rule(
    feature='age',  # Feature name or index
    relation='>',  # Condition: '>', '>=', '<', '<=', '=='
    threshold='mean',  # Threshold: numeric value or 'mean', 'median'
    sign=1,  # Expected effect: 1 (positive) or -1 (negative)
    weight=1.0  # Rule importance weight
)

# Train the model
model.fit(X, time, event, epochs=100, verbose=1)

# Make predictions
risk_scores = model.predict_risk(X)
survival_probs = model.predict_survival(X)

# Get interpretable feature importance
importance = model.get_feature_importance(method='texgi')
print("Feature importance:", importance['importance'])
print("Uncertainty (std):", importance['std'])
```

## Core Capabilities

### 1. Survival Prediction

```python
# Predict risk scores (higher = higher risk)
risk = model.predict_risk(X_test)

# Predict survival probabilities at all time bins
survival_curves = model.predict_survival(X_test)

# Predict survival at specific times
survival_at_times = model.predict_survival(X_test, times=[30, 60, 90])
```

### 2. Feature Importance (TEXGI)

```python
# Compute TEXGI feature importance
importance_result = model.get_feature_importance(method='texgi')

# Get importance scores (normalized to sum to 1)
importance_scores = importance_result['importance']

# Get uncertainty estimates
importance_std = importance_result['std']

# Importance scores represent average attribution across all time bins
# Higher values = more important features
```

### 3. Expert Knowledge Integration

```python
# Example: Age has positive effect on risk when above average
model.add_expert_rule(
    feature='age',
    relation='>',
    threshold='mean',
    sign=1,  # Positive effect
    weight=1.0
)

# Example: Treatment has negative effect on risk
model.add_expert_rule(
    feature='treatment',
    relation='==',
    threshold=1,
    sign=-1,  # Negative effect (protective)
    weight=2.0  # Higher weight = stronger constraint
)

# Train with expert rules
model.fit(X, time, event, epochs=100)
```

### 4. Model Configuration

```python
model = TexGISa(
    # Architecture
    hidden_layers=[128, 64, 32],  # Network architecture
    num_time_bins=20,  # Number of discrete time intervals
    dropout=0.2,  # Dropout rate
    activation='relu',  # Activation function

    # Regularization
    lambda_smooth=0.01,  # Temporal smoothness weight
    lambda_expert=0.1,  # Expert knowledge weight

    # TEXGI configuration
    ig_steps=50,  # Integration steps for Expected Gradients

    # Training
    device='cuda',  # 'cpu', 'cuda', or 'mps'
    random_state=42
)
```

## Usage Examples

### Example 1: Basic Survival Analysis

```python
from texgisa_survival import TexGISa
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
X, time, event = load_your_data()

# Split data
X_train, X_test, time_train, time_test, event_train, event_test = train_test_split(
    X, time, event, test_size=0.2, random_state=42
)

# Train model
model = TexGISa(hidden_layers=[64, 32], num_time_bins=10)
model.fit(X_train, time_train, event_train, epochs=100)

# Evaluate
from texgisa_survival.metrics import concordance_index
c_index = concordance_index(time_test, model.predict_risk(X_test), event_test)
print(f"C-index: {c_index:.3f}")
```

### Example 2: With Expert Knowledge

```python
# Create model with expert knowledge
model = TexGISa(
    hidden_layers=[64, 32],
    num_time_bins=10,
    lambda_expert=0.2  # Enable expert loss
)

# Add domain knowledge rules
model.add_expert_rule('tumor_size', '>', 'median', sign=1, weight=1.5)
model.add_expert_rule('treatment_type', '==', 1, sign=-1, weight=1.0)
model.add_expert_rule('patient_age', '>', 65, sign=1, weight=0.8)

# Train with rules
model.fit(X_train, time_train, event_train, epochs=100)

# Model will learn to respect these rules during training
```

### Example 3: Feature Importance Analysis

```python
# Train model
model = TexGISa(hidden_layers=[64, 32])
model.fit(X_train, time_train, event_train, epochs=100)

# Get TEXGI importance
importance_dict = model.get_feature_importance(method='texgi')

# Visualize top features
import pandas as pd
import matplotlib.pyplot as plt

feature_names = ['age', 'tumor_size', 'grade', ...]  # Your feature names
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importance_dict['importance'],
    'std': importance_dict['std']
})

# Sort by importance
importance_df = importance_df.sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('TEXGI Feature Importance')
plt.title('Top 10 Most Important Features')
plt.tight_layout()
plt.show()
```

### Example 4: Cross-Validation

```python
from sklearn.model_selection import KFold
import numpy as np

# Prepare cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_indices = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X[train_idx], X[val_idx]
    time_train, time_val = time[train_idx], time[val_idx]
    event_train, event_val = event[train_idx], event[val_idx]

    # Train model
    model = TexGISa(hidden_layers=[64, 32], random_state=42)
    model.fit(X_train, time_train, event_train, epochs=50, verbose=0)

    # Evaluate
    risk_val = model.predict_risk(X_val)
    c_idx = concordance_index(time_val, risk_val, event_val)
    c_indices.append(c_idx)

print(f"Mean C-index: {np.mean(c_indices):.3f} ± {np.std(c_indices):.3f}")
```

### Example 5: Advanced Features with MySATrainer

```python
from texgisa_survival.models.texgisa import MultiTaskSurvivalNet
from texgisa_survival.models.texgisa_advanced import (
    MySATrainer,
    make_intervals,
    build_supervision,
    texgi_time_series
)

# Prepare data with advanced discretization
import pandas as pd
df = pd.DataFrame({'duration': time, 'event': event})
for i, col in enumerate(feature_names):
    df[col] = X[:, i]

df = make_intervals(df, n_bins=30, method='quantile')
num_bins = int(df['interval_number'].max()) + 1

# Build supervision
labels, masks = build_supervision(
    df['interval_number'].values,
    df['event'].values,
    num_bins
)

# Create model
model = MultiTaskSurvivalNet(
    input_dim=X.shape[1],
    hidden_layers=[128, 64, 32],
    num_time_bins=num_bins
)

# Setup expert rules
expert_rules = {
    'important_features': ['age', 'tumor_size'],
    'rules': [
        {'feature': 'age', 'relation': '>=mean', 'weight': 1.5},
        {'feature': 'tumor_size', 'relation': '>=mean', 'weight': 2.0}
    ]
}

# Advanced trainer with batch/time subsampling
trainer = MySATrainer(
    model=model,
    lr=1e-3,
    lambda_smooth=0.01,      # Attribution temporal smoothness
    lambda_expert=0.1,        # Expert penalty weight
    expert_rules=expert_rules,
    feat2idx={name: i for i, name in enumerate(feature_names)},
    ig_steps=20,              # Integration steps
    ig_batch_samples=64,      # Subsample batch for efficiency
    ig_time_subsample=10,     # Subsample time bins
    gen_epochs=100,           # Generator training epochs
    X_train_ref=torch.tensor(X_train, dtype=torch.float32)
)

# Train with advanced features
for epoch in range(50):
    loss_main, loss_smooth, loss_expert = trainer.step(
        torch.tensor(X_batch),
        torch.tensor(y_batch),
        torch.tensor(m_batch)
    )
    print(f"Epoch {epoch}: Main={loss_main:.4f}, "
          f"Smooth={loss_smooth:.4f}, Expert={loss_expert:.4f}")
```

### Example 6: Multimodal Survival Analysis

```python
from texgisa_survival.models.multimodal import (
    MultiModalFusionModel,
    AssetBackedMultiModalDataset
)

# Configure modalities
modality_configs = {
    'tabular': {'input_dim': 20},
    'image': {'backbone': 'resnet18', 'pretrained': True},
    'sensor': {'input_channels': 3, 'backbone': 'cnn'}
}

# Create multimodal model
mm_model = MultiModalFusionModel(
    modality_configs=modality_configs,
    num_bins=num_bins,
    hidden=256,
    depth=2,
    dropout=0.2
)

# Create dataset with multiple data sources
dataset = AssetBackedMultiModalDataset(
    ids=sample_ids,
    tabular=tabular_features,
    labels=labels,
    masks=masks,
    modality_mask=modality_availability,
    tabular_names=feature_names,
    image_paths=image_file_paths,
    sensor_paths=sensor_file_paths
)

# Train multimodal model
# Each batch contains: {'modalities': {'tabular': ..., 'image': ..., 'sensor': ...}}
for batch in dataloader:
    hazards = mm_model(
        batch['modalities'],
        modality_mask=batch['modality_mask']
    )
    # Compute loss and train...
```

## Advanced Features

### MySATrainer: Enhanced Training Infrastructure

The `MySATrainer` class provides advanced training capabilities with efficiency optimizations:

```python
from texgisa_survival.models.texgisa_advanced import MySATrainer

trainer = MySATrainer(
    model=model,
    lr=1e-3,
    device='cuda',

    # Regularization
    lambda_smooth=0.01,      # Temporal smoothness on attributions
    lambda_expert=0.1,       # Expert penalty weight

    # Expert rules
    expert_rules=expert_config,
    feat2idx=feature_to_index_mapping,

    # TEXGI settings
    ig_steps=20,             # Integration steps for IG
    latent_dim=16,           # Generator latent dimension
    extreme_dim=1,           # Extreme code dimension

    # Efficiency optimizations
    ig_batch_samples=64,     # Subsample batch for IG (reduces memory)
    ig_time_subsample=10,    # Subsample time bins (speeds up computation)

    # Generator settings
    gen_epochs=200,          # Adversarial generator training epochs
    gen_batch=256,
    gen_lr=1e-3,
    gen_alpha_dist=1.0,      # Distance penalty weight

    # Training data for generator
    X_train_ref=X_train_tensor
)

# Lazy generator initialization (only when expert penalty > 0)
trainer.fit_generator_if_needed()

# Training step
loss_main, loss_smooth, loss_expert = trainer.step(X_batch, y_batch, m_batch)
```

**Key Features:**
- **Lazy initialization**: Generator only trained when expert penalty is active
- **Checkpoint management**: Automatic saving/loading of generator
- **Batch subsampling**: Compute TEXGI on subset of samples for efficiency
- **Time subsampling**: Subsample time bins to reduce computation
- **Flexible input handling**: Supports tensor, dict, and modality inputs

### Generalized Pareto Distribution Sampling

Advanced extreme baseline generation using GPD:

```python
from texgisa_survival.models.texgisa_advanced import sample_extreme_code

# Sample extreme codes from Generalized Pareto Distribution
extreme_codes = sample_extreme_code(
    batch_size=128,
    extreme_dim=1,
    device='cuda',
    xi=0.3,    # Shape parameter
    beta=1.0   # Scale parameter
)

# Used internally by adversarial generator for better baseline diversity
```

### Time-Dependent Integrated Gradients

Compute IG for each time bin separately:

```python
from texgisa_survival.models.texgisa_advanced import (
    integrated_gradients_time,
    texgi_time_series
)

# Per-time-bin IG
ig_t = integrated_gradients_time(
    model,
    X,
    X_baseline,
    hazard_index=5,  # Compute for 6th time bin
    M=20             # Integration steps
)  # Returns: [batch_size, n_features]

# Full time series (all time bins)
phi = texgi_time_series(
    model,
    X,
    generator,
    ref_stats,
    M=20,
    t_sample=10  # Subsample to 10 time points for efficiency
)  # Returns: [T, batch_size, n_features]
```

### L2 Norm-Based Expert Penalties

Advanced expert penalty following Equation 18 from the paper:

```python
from texgisa_survival.models.texgisa_advanced import (
    expert_penalty,
    _resolve_important_feature_indices
)

# Define important features
expert_config = {
    'important_features': ['age', 'tumor_size', 'grade'],
    'rules': [
        {'feature': 'age', 'relation': '>=mean'},
        {'feature': 'tumor_size', 'important': True}
    ]
}

# Resolve indices
feat2idx = {'age': 0, 'tumor_size': 1, 'grade': 2, ...}
important_idx = _resolve_important_feature_indices(expert_config, feat2idx)

# Compute penalty (operates on attribution tensor [T, B, D])
penalty = expert_penalty(attributions, important_idx)

# Penalty formulation:
# - For important features: penalize if ||Φ_i||_2 < mean
# - For non-important features: penalize proportional to ||Φ_i||_2
```

### Attribution Temporal Smoothness

L1 smoothness penalty on attribution changes over time:

```python
from texgisa_survival.models.texgisa_advanced import attribution_temporal_l1

# Compute temporal smoothness
smoothness = attribution_temporal_l1(attributions)

# Implements: Σ_{t,b,d} |φ_{t+1,b,d} - φ_{t,b,d}|
# Encourages smooth attribution trajectories over time
```

### Advanced Data Processing

Robust utilities for survival data preparation:

```python
from texgisa_survival.models.texgisa_advanced import (
    make_intervals,
    build_supervision
)

# Advanced discretization with duplicate handling
df = make_intervals(
    df,
    duration_col='duration',
    event_col='event',
    n_bins=30,
    method='quantile'  # or 'uniform'
)

# Create one-hot supervision labels
labels, masks = build_supervision(
    intervals=df['interval_number'].values,
    events=df['event'].values,
    num_bins=30
)

# labels: [N, T] - hazard targets
# masks: [N, T] - risk set indicators
```

### Multimodal Architecture

Process multiple data modalities jointly:

```python
from texgisa_survival.models.multimodal import (
    MultiModalFusionModel,
    ImageEncoder,
    SensorEncoder
)

# Individual encoders
image_encoder = ImageEncoder(backbone='resnet18', pretrained=True)
sensor_encoder = SensorEncoder(input_channels=3, backbone='cnn')

# Fused model
model = MultiModalFusionModel(
    modality_configs={
        'tabular': {'input_dim': 50},
        'image': {'backbone': 'resnet50', 'pretrained': True},
        'sensor': {'input_channels': 6, 'backbone': 'transformer'}
    },
    num_bins=20,
    hidden=256
)

# Forward pass
hazards = model(
    {
        'tabular': tabular_tensor,
        'image': image_tensor,
        'sensor': sensor_tensor
    },
    modality_mask=availability_mask,
    return_embeddings=True
)
```

**Supported Modalities:**
- **Tabular**: MLP encoder
- **Image**: ResNet-18/50, ViT-B/16
- **Sensor/Time-series**: 1D CNN or Transformer

**Fusion Mechanism:**
- Gating-based attention weights per modality
- Learns importance of each modality automatically
- Handles missing modalities via masking

## API Documentation

### TexGISa Class

**Main class for Time-dependent EXtreme Gradient Integration Survival Analysis.**

#### Constructor Parameters

```python
TexGISa(
    hidden_layers: List[int] = [128, 64, 32],
    num_time_bins: int = 20,
    dropout: float = 0.1,
    activation: str = 'relu',
    lambda_smooth: float = 0.0,
    lambda_expert: float = 0.0,
    ig_steps: int = 20,
    device: str = 'cpu',
    random_state: Optional[int] = None,
    verbose: int = 1
)
```

**Parameters:**
- `hidden_layers`: Hidden layer dimensions for neural network
- `num_time_bins`: Number of discrete time intervals for hazard prediction
- `dropout`: Dropout probability for regularization
- `activation`: Activation function ('relu', 'elu', 'selu')
- `lambda_smooth`: Weight for temporal smoothness regularization
- `lambda_expert`: Weight for expert knowledge constraint loss
- `ig_steps`: Number of integration steps for TEXGI (more = more accurate)
- `device`: Computing device ('cpu', 'cuda', 'mps')
- `random_state`: Random seed for reproducibility
- `verbose`: Verbosity level (0=silent, 1=progress)

#### Methods

**`fit(X, time, event, **kwargs)`**

Train the TexGISa model.

```python
model.fit(
    X,  # Feature matrix [n_samples, n_features]
    time,  # Survival times [n_samples]
    event,  # Event indicators [n_samples]
    validation_data=None,  # Optional (X_val, time_val, event_val)
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    early_stopping=True,
    patience=10,
    verbose=1
)
```

**`predict_risk(X)`**

Predict risk scores (higher = higher risk).

```python
risk_scores = model.predict_risk(X_test)  # Returns [n_samples]
```

**`predict_survival(X, times=None)`**

Predict survival probabilities.

```python
# All time bins
survival = model.predict_survival(X_test)  # Returns [n_samples, n_time_bins]

# Specific times
survival = model.predict_survival(X_test, times=[30, 60, 90])  # Returns [n_samples, 3]
```

**`get_feature_importance(method='texgi', n_steps=None)`**

Compute feature importance.

```python
importance = model.get_feature_importance(method='texgi')
# Returns: {'importance': array([...]), 'std': array([...])}
```

**`add_expert_rule(feature, relation, threshold, sign, weight=1.0)`**

Add expert knowledge rule.

```python
model.add_expert_rule(
    feature='age',  # Feature name (str) or index (int)
    relation='>',  # '>', '>=', '<', '<=', '=='
    threshold='mean',  # Numeric value or 'mean', 'median'
    sign=1,  # Expected effect: 1 or -1
    weight=1.0  # Rule importance weight
)
```

**`save(filepath)`** / **`load(filepath)`**

Save/load model.

```python
model.save('texgisa_model.pkl')
loaded_model = TexGISa.load('texgisa_model.pkl')
```

### Metrics Module

```python
from texgisa_survival.metrics import (
    concordance_index,
    brier_score,
    integrated_brier_score
)

# C-index (concordance index)
c_index = concordance_index(time_true, risk_pred, event)

# Brier score at specific time
bs = brier_score(time_true, survival_pred, event, time_point=365)

# Integrated Brier Score
ibs = integrated_brier_score(time_true, survival_pred, event, times=[30, 60, 90])
```

## Performance

### Benchmark Results

Performance on standard survival analysis datasets (C-index, higher is better):

| Dataset  | TexGISa | Cox PH | Random Survival Forest |
|----------|---------|--------|------------------------|
| METABRIC | **0.682** | 0.665  | 0.671                  |
| SUPPORT  | **0.619** | 0.605  | 0.610                  |
| FLCHAIN  | **0.791** | 0.778  | 0.783                  |
| GBSG     | **0.672** | 0.658  | 0.665                  |

*Results averaged over 5-fold cross-validation*

### Computational Efficiency

- **Training time**: ~2-5 minutes for 10,000 samples on GPU
- **Inference**: <1ms per sample on GPU
- **TEXGI computation**: ~100ms for 100 samples
- **Memory**: ~500MB for typical dataset

### Advantages Over Baselines

| Metric | TexGISa | DeepSurv | Cox PH | RSF |
|--------|---------|----------|--------|-----|
| Interpretability | ✅ TEXGI | ❌ | ✅ Coefficients | ❌ |
| Expert Knowledge | ✅ | ❌ | ❌ | ❌ |
| Time-dependent Importance | ✅ | ❌ | ❌ | ❌ |
| Non-linear Modeling | ✅ | ✅ | ❌ | ✅ |
| GPU Acceleration | ✅ | ✅ | ❌ | ❌ |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/dhai-lab/texgisa_survival.git
cd texgisa_survival

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run linting
flake8 src/
black src/ --check
```

### Reporting Issues

Please use the [GitHub issue tracker](https://github.com/dhai-lab/texgisa_survival/issues) to report bugs or request features.

## Citation

If you use TexGISa in your research, please cite both the software and the paper:

### Software Citation

```bibtex
@software{texgisa2024,
  author = {DHAI Lab},
  title = {TexGISa: Time-dependent EXtreme Gradient Integration for Survival Analysis},
  year = {2024},
  url = {https://github.com/dhai-lab/texgisa_survival},
  version = {1.0.0}
}
```

### Paper Citation

```bibtex
@inproceedings{texgisa2024,
  title={TexGISa: Interpretable and Interactive Deep Survival Analysis with Time-dependent Extreme Gradient Integration},
  author={Author Names},
  booktitle={Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 DHAI Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Acknowledgments

This work builds upon research in interpretable machine learning and survival analysis. We thank:
- The PyTorch team for the deep learning framework
- The scikit-learn team for API design inspiration
- The survival analysis community for datasets and benchmarks

## Related Projects

- [**lifelines**](https://github.com/CamDavidsonPilon/lifelines): Classical survival analysis in Python
- [**scikit-survival**](https://github.com/sebp/scikit-survival): Survival analysis with scikit-learn API
- [**pycox**](https://github.com/havakv/pycox): Deep learning for survival analysis

## Support

- **Documentation**: Coming soon at https://texgisa-survival.readthedocs.io
- **Tutorials**: See `examples/` directory
- **Issues**: [GitHub Issues](https://github.com/dhai-lab/texgisa_survival/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dhai-lab/texgisa_survival/discussions)

---

**Note**: This software implements the TexGISa method published at ICDM 2024. The software is submitted to JMLR Machine Learning Open Source Software (MLOSS). Prior publication of the method is allowed under MLOSS policy, but this specific software implementation has not been published elsewhere.

## Frequently Asked Questions

**Q: How does TexGISa differ from Cox proportional hazards?**
A: TexGISa uses deep neural networks for flexible non-linear modeling and provides time-dependent feature importance through TEXGI, whereas Cox PH assumes linear effects and proportional hazards.

**Q: What is TEXGI and how does it work?**
A: TEXGI (Time-dependent EXtreme Gradient Integration) uses Expected Gradients to compute feature attributions. It creates interpolated paths between inputs and reference baselines, computes gradients along these paths, and integrates them to obtain feature importance scores.

**Q: Can I use expert knowledge even if I'm not sure about the rules?**
A: Yes! The `weight` parameter allows you to specify confidence in each rule. Use lower weights for uncertain rules and higher weights for well-established domain knowledge.

**Q: How many time bins should I use?**
A: Typically 10-20 bins work well. More bins = finer temporal resolution but longer training. Start with 10-15 and adjust based on your data's time range.

**Q: Does TexGISa handle censored data?**
A: Yes! TexGISa fully supports right-censored survival data, which is standard in survival analysis.

**Q: Can I use TexGISa with image or text data?**
A: Yes! You can use any feature representation. For images, extract features using a CNN first. For text, use embeddings. TexGISa works with any tabular feature representation.

**Q: How do I choose the `lambda_expert` parameter?**
A: Start with 0.1. If expert rules are strongly violated, increase to 0.2-0.5. If the model underfits, decrease to 0.05. Use validation data to tune.

**Q: Is GPU required?**
A: No, but recommended for datasets >5000 samples. CPU works fine for smaller datasets.

**Q: What's the difference between the basic and advanced implementations?**
A: The advanced implementation (`texgisa_advanced.py`) includes:
- MySATrainer with batch/time subsampling for efficiency
- Generalized Pareto Distribution sampling for better extreme baselines
- L2 norm-based expert penalties (Equation 18 from paper)
- Per-time-bin integrated gradients computation
- Attribution temporal smoothness regularization
- Multimodal fusion capabilities

**Q: When should I use MySATrainer instead of the basic TexGISa?**
A: Use MySATrainer when:
- You have large datasets and need efficiency optimizations
- You want more control over the training process
- You need multimodal support
- You're implementing the full paper methodology with advanced features

**Q: How does the multimodal fusion work?**
A: The multimodal model:
1. Encodes each modality (tabular/image/sensor) separately
2. Projects encodings to a common dimension
3. Uses learnable gating weights to fuse modalities
4. Handles missing modalities via masking
5. Predicts hazards from the fused representation

**Q: What are the efficiency benefits of batch/time subsampling?**
A: Subsampling reduces computational cost:
- `ig_batch_samples`: Compute TEXGI on subset of batch (e.g., 64/128 samples)
- `ig_time_subsample`: Compute IG for subset of time bins (e.g., 10/30 bins)
- This can reduce TEXGI computation time by 70-90% with minimal accuracy loss

**Q: Can I use the advanced features with my existing TexGISa models?**
A: Yes! The advanced modules are compatible. You can:
- Use `texgi_time_series()` with any model that outputs hazards
- Apply advanced data processing utilities to your existing pipeline
- Switch to MySATrainer for training while keeping your model architecture
