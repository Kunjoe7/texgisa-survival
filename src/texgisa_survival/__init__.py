# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
TexGISa: Time-dependent EXtreme Gradient Integration for Survival Analysis

An interpretable deep learning model for survival analysis with expert
knowledge integration and time-dependent feature importance.
"""

from .api import TexGISa, load_model
from .data import SurvivalDataset
from .metrics import concordance_index, brier_score, integrated_brier_score

# Optional preprocessing modules
try:
    from .preprocessing import TabularProcessor
    _has_preprocessing = True
except ImportError:
    _has_preprocessing = False

__version__ = "1.0.0"

__all__ = [
    "TexGISa",
    "load_model",
    "SurvivalDataset",
    "concordance_index",
    "brier_score",
    "integrated_brier_score",
]

if _has_preprocessing:
    __all__.append("TabularProcessor")