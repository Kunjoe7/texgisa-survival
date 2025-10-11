# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Preprocessing utilities for various data modalities in survival analysis.

This module provides specialized preprocessing for:
- Sensor/time-series data
- Medical imaging data
- Tabular data transformations
"""

from typing import Optional, List, Dict, Any
import warnings

# Try to import specialized preprocessors
try:
    from .sensor import SensorProcessor
    _has_sensor = True
except ImportError as e:
    _has_sensor = False
    _sensor_error = str(e)

try:
    from .image import ImageProcessor
    _has_image = True
except ImportError as e:
    _has_image = False
    _image_error = str(e)

# Always available
from .tabular import TabularProcessor


def get_sensor_processor(*args, **kwargs):
    """Get sensor processor with helpful error if dependencies missing."""
    if not _has_sensor:
        raise ImportError(
            f"Sensor processing requires additional dependencies. "
            f"Install with: pip install texgisa-survival[sensors]\n"
            f"Original error: {_sensor_error}"
        )
    return SensorProcessor(*args, **kwargs)


def get_image_processor(*args, **kwargs):
    """Get image processor with helpful error if dependencies missing."""
    if not _has_image:
        raise ImportError(
            f"Image processing requires additional dependencies. "
            f"Install with: pip install texgisa-survival[images]\n"
            f"Original error: {_image_error}"
        )
    return ImageProcessor(*args, **kwargs)


__all__ = [
    'TabularProcessor',
    'get_sensor_processor',
    'get_image_processor'
]

# Conditional exports
if _has_sensor:
    __all__.append('SensorProcessor')
if _has_image:
    __all__.append('ImageProcessor')