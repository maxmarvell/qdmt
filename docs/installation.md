# Installation Guide

## Requirements

- Python >= 3.11
- pip or conda

## Installation Methods

### From Source (Recommended for Development)

```bash
git clone https://github.com/yourusername/qdmt.git
cd qdmt
pip install -e .
```

### Using pip (when published)

```bash
pip install qdmt
```

### Using conda

```bash
conda env create -f environment.yml
conda activate qdmt
```

## Optional Dependencies

### Development Tools

For development, install additional dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-benchmark: Performance benchmarking
- ruff: Linting and formatting
- mypy: Type checking
- pre-commit: Git hooks

### Documentation

To build documentation locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

### Jupyter Notebooks

For running example notebooks:

```bash
pip install -e ".[notebooks]"
```

### All Optional Dependencies

Install everything at once:

```bash
pip install -e ".[all]"
```

## Verifying Installation

Test your installation:

```python
import qdmt
from qdmt.uniform_mps import UniformMps

# Create a random MPS
A = UniformMps.random(D=5, d=2)
print(f"Created MPS with shape: {A.tensor.shape}")
```

Run the test suite:

```bash
pytest
```

## Troubleshooting

### Common Issues

**Import errors**: Make sure you've installed all required dependencies:
```bash
pip install numpy scipy ncon matplotlib opt_einsum
```

**Test failures**: Ensure you're using Python 3.11 or higher:
```bash
python --version
```

### Getting Help

If you encounter issues:
1. Check the [FAQ](faq.md)
2. Search [existing issues](https://github.com/yourusername/qdmt/issues)
3. Open a new issue with details about your environment
