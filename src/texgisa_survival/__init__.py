# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
TexGISa: Time-dependent EXtreme Gradient Integration for Survival Analysis

An interpretable deep learning model for survival analysis with expert
knowledge integration and time-dependent feature importance.
"""

from .api import TexGISa, load_model
from .data import SurvivalDataset, DataManager
from .metrics import concordance_index, brier_score, integrated_brier_score, kaplan_meier_estimator
from .diagnostics import (
    brier_score_discrete,
    calibration_by_quantile,
    hazards_to_survival,
    integrated_brier_score_discrete,
    integrated_nll_discrete,
    compute_diagnostics,
    plot_diagnostics,
)
from .texgi_utils import (
    sample_extreme_code,
    attribution_temporal_l1,
    aggregate_importance,
    expert_penalty,
    masked_bce_nll,
    topk_feature_importance,
    integrated_gradients_time,
    texgi_time_series,
    standardize_features,
    destandardize_features,
)
from .explainers.counterfactual import (
    CounterfactualExplainer,
    CFResult,
    TexgiCFResult,
    generate_cf_from_arrays,
    generate_texgi_counterfactuals,
)

# Optional baseline models (require pycox)
try:
    from .models.baselines import CoxTimeModel, DeepSurvModel, DeepHitModel
    _has_baselines = True
except ImportError:
    _has_baselines = False

# Optional preprocessing modules
try:
    from .preprocessing import TabularProcessor
    _has_preprocessing = True
except ImportError:
    _has_preprocessing = False

__version__ = "2.0.0"

__all__ = [
    # Main API
    "TexGISa",
    "load_model",
    "SurvivalDataset",
    "DataManager",
    # Metrics
    "concordance_index",
    "brier_score",
    "integrated_brier_score",
    "kaplan_meier_estimator",
    # Diagnostics
    "brier_score_discrete",
    "calibration_by_quantile",
    "hazards_to_survival",
    "integrated_brier_score_discrete",
    "integrated_nll_discrete",
    "compute_diagnostics",
    "plot_diagnostics",
    # TEXGI utilities
    "sample_extreme_code",
    "attribution_temporal_l1",
    "aggregate_importance",
    "expert_penalty",
    "masked_bce_nll",
    "topk_feature_importance",
    "integrated_gradients_time",
    "texgi_time_series",
    "standardize_features",
    "destandardize_features",
    # Counterfactual explanations
    "CounterfactualExplainer",
    "CFResult",
    "TexgiCFResult",
    "generate_cf_from_arrays",
    "generate_texgi_counterfactuals",
]

if _has_baselines:
    __all__.extend(["CoxTimeModel", "DeepSurvModel", "DeepHitModel"])

if _has_preprocessing:
    __all__.append("TabularProcessor")