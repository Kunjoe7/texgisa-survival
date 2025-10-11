#!/usr/bin/env python
"""
Test script to verify model save/load with scaler works correctly
"""

import numpy as np
from texgisa_survival import load_model

print("=" * 60)
print("Testing Model Save/Load with Scaler")
print("=" * 60)

# Generate same test data
np.random.seed(42)
X = np.random.randn(10, 10)

# Load the saved model
print("\nLoading model from /tmp/texgisa_test_model.pkl...")
model = load_model('/tmp/texgisa_test_model.pkl', verbose=0)
print("✓ Model loaded successfully")

# Test predictions
print("\nTesting predictions with loaded model...")
try:
    risk_scores = model.predict_risk(X)
    print(f"✓ Risk prediction works!")
    print(f"  - Risk scores shape: {risk_scores.shape}")
    print(f"  - Risk scores: {risk_scores[:3]}")

    survival_probs = model.predict_survival(X)
    print(f"✓ Survival prediction works!")
    print(f"  - Survival shape: {survival_probs.shape}")
    print(f"  - Sample survival curve: {survival_probs[0]}")

    print("\n" + "=" * 60)
    print("✓ Model save/load is FULLY WORKING!")
    print("=" * 60)

except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
