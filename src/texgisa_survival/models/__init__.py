# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
TexGISa survival analysis model.

This module provides the TexGISa (Time-dependent EXtreme Gradient Integration
for Survival Analysis) model - an interpretable deep learning approach for
survival analysis with expert knowledge integration.
"""

from .base import BaseSurvivalModel
from .texgisa import TEXGISAModel

__all__ = [
    'BaseSurvivalModel',
    'TEXGISAModel'
]