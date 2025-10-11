#!/usr/bin/env python
# Copyright (c) 2024 DHAI Lab
# Licensed under the MIT License - see LICENSE file for details

"""
Setup script for DHAI Survival Analysis Package

This package provides comprehensive deep learning tools for survival analysis
with interpretable AI capabilities, including implementations of Cox regression,
DeepSurv, DeepHit, and TexGISa models.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements_file = this_directory / "requirements.txt"
install_requires = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    # Fallback requirements matching pyproject.toml
    install_requires = [
        "numpy>=1.19.0,<2.0.0",
        "pandas>=1.2.0",
        "scikit-learn>=0.24.0",
        "torch>=1.9.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "lifelines>=0.25.0",
        "pycox>=0.2.0",
        "torchtuples>=0.2.0",
        "joblib>=1.0.0",
        "tqdm>=4.60.0",
        "scipy>=1.6.0",
    ]

setup(
    name="texgisa-survival",  # Using hyphenated name to match pyproject.toml
    version="1.0.0",       # Updated to match pyproject.toml
    author="DHAI Lab",
    author_email="texgisa-survival@example.com",  # Match pyproject.toml
    maintainer="DHAI Lab",
    maintainer_email="texgisa-survival@example.com",
    description="A comprehensive deep learning library for survival analysis with interpretable AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dhai-lab/texgisa_survival",
    project_urls={
        "Bug Reports": "https://github.com/dhai-lab/texgisa_survival/issues",
        "Source": "https://github.com/dhai-lab/texgisa_survival",
        "Documentation": "https://texgisa-survival.readthedocs.io",
        "Changelog": "https://github.com/dhai-lab/texgisa_survival/blob/main/CHANGELOG.md",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",  # Updated to match pyproject.toml
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    keywords=[
        "survival-analysis",
        "deep-learning", 
        "machine-learning",
        "interpretable-ai",
        "time-to-event",
        "cox-regression",
        "pytorch",
        "healthcare",
        "biostatistics",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
            "coverage[toml]>=6.5",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "nbsphinx>=0.8.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
        ],
        "examples": [
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
            "ipywidgets>=7.6.0",
        ],
        "all": [
            "texgisa-survival[dev,docs,examples]",
        ],
    },
    entry_points={
        "console_scripts": [
            "texgisa-survival=texgisa_survival.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    # Additional metadata for academic/research software
    license="MIT",
    platforms=["any"],
)