# Contributing to DHAI Survival

First off, thank you for considering contributing to DHAI Survival! It's people like you that make DHAI Survival such a great tool for the survival analysis community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Commit Messages](#commit-messages)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to texgisa-survival@example.com.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Push to your fork and submit a pull request

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** to demonstrate the steps
- **Describe the behavior you observed** and explain why it's a problem
- **Explain the behavior you expected** to see instead
- **Include screenshots** if relevant
- **Include your environment details** (Python version, OS, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Provide specific examples** to demonstrate the enhancement
- **Describe the current behavior** and explain why it's insufficient
- **Explain why this enhancement would be useful** to most users
- **List any alternatives** you've considered

### Code Contributions

#### Your First Code Contribution

Unsure where to begin? You can start by looking through these issues:

- Issues labeled `good first issue` - issues which should be relatively simple to implement
- Issues labeled `help wanted` - issues which need extra attention
- Issues labeled `documentation` - improvements or additions to documentation

#### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/texgisa_survival.git
   cd texgisa_survival
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Create a branch for your feature**
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Adding New Models

If you're adding a new survival model:

1. Create a new module in `src/texgisa_survival/models/`
2. Inherit from the `BaseModel` class
3. Implement required methods: `fit()`, `predict_risk()`, etc.
4. Add comprehensive tests in `tests/test_models.py`
5. Add documentation with examples
6. Update the README with model description

Example structure:
```python
from texgisa_survival.models.base import BaseModel

class YourModel(BaseModel):
    """
    Your model description.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    **kwargs
        Additional parameters
    
    Examples
    --------
    >>> model = YourModel(input_dim=10)
    >>> model.fit(X_train, y_train, e_train)
    >>> risk_scores = model.predict_risk(X_test)
    """
    
    def __init__(self, input_dim, **kwargs):
        super().__init__(input_dim, **kwargs)
        # Your initialization code
    
    def fit(self, X, y, e, **kwargs):
        """Fit the model"""
        # Your training code
    
    def predict_risk(self, X):
        """Predict risk scores"""
        # Your prediction code
```

## Style Guidelines

### Python Style Guide

We use [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following specifications:

- Line length: 100 characters
- Use type hints where possible
- Document all public functions with NumPy-style docstrings

We enforce style using:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `isort` for import sorting

Run these tools before committing:
```bash
black src/
flake8 src/
mypy src/texgisa_survival
isort src/
```

Or use pre-commit:
```bash
pre-commit run --all-files
```

### Documentation Style

- Use NumPy style for docstrings
- Include type hints in function signatures
- Provide examples in docstrings where helpful
- Keep documentation up-to-date with code changes

Example docstring:
```python
def concordance_index(y_true, y_pred, e):
    """
    Calculate the concordance index for survival analysis.
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        True survival times
    y_pred : array-like, shape = [n_samples]
        Predicted risk scores
    e : array-like, shape = [n_samples]
        Event indicators (1 if event occurred, 0 if censored)
    
    Returns
    -------
    float
        Concordance index between 0 and 1
    
    Examples
    --------
    >>> c_index = concordance_index([1, 2, 3], [0.5, 0.3, 0.1], [1, 1, 0])
    >>> print(f"C-index: {c_index:.3f}")
    C-index: 1.000
    """
```

## Testing

### Writing Tests

- Write tests for any new functionality
- Ensure all tests pass before submitting PR
- Aim for high test coverage (>90%)
- Use pytest for testing

Run tests:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=texgisa_survival --cov-report=html
```

### Test Structure

```python
import pytest
from texgisa_survival.your_module import your_function

class TestYourFunction:
    def test_normal_case(self):
        """Test normal operation"""
        result = your_function(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge cases"""
        with pytest.raises(ValueError):
            your_function(invalid_input)
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_cases(self, input, expected):
        """Test multiple cases"""
        assert your_function(input) == expected
```

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` A new feature
- `fix:` A bug fix
- `docs:` Documentation only changes
- `style:` Changes that don't affect code meaning
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `perf:` Code change that improves performance
- `test:` Adding missing tests or correcting existing tests
- `build:` Changes that affect the build system or dependencies
- `ci:` Changes to CI configuration files and scripts
- `chore:` Other changes that don't modify src or test files

Examples:
```
feat: add TexGISa model for interpretable survival analysis
fix: correct concordance index calculation for tied events
docs: update installation instructions for Windows users
test: add integration tests for model comparison
```

## Pull Request Process

1. **Ensure your code follows the style guidelines**
2. **Update documentation** for any changed functionality
3. **Add tests** for new functionality
4. **Ensure all tests pass** locally
5. **Update the CHANGELOG.md** with your changes
6. **Create a pull request** with a clear title and description

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Commit messages follow conventions
- [ ] PR has descriptive title and description
- [ ] All CI checks passing

### PR Review Process

1. At least one maintainer review required
2. All CI checks must pass
3. No merge conflicts
4. Discussions resolved

## Release Process

Releases are managed by maintainers following semantic versioning:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. GitHub Actions automatically publishes to PyPI

## Community

### Getting Help

- GitHub Issues for bug reports and feature requests
- GitHub Discussions for general questions
- Email: texgisa-survival@example.com

### Acknowledgments

Contributors will be acknowledged in:
- The AUTHORS file
- Release notes
- Project documentation

## Questions?

Feel free to contact the maintainers if you have any questions about contributing. We're here to help and look forward to your contributions!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.