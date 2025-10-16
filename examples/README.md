# Examples

This directory contains examples demonstrating how to use QDMT.

## Structure

- `scripts/`: Python scripts showing various use cases
- `notebooks/`: Jupyter notebooks with interactive tutorials

## Quick Start Scripts

### Basic Time Evolution

See `scripts/basic_evolution.py` for a minimal example of time evolution using the Transverse Field Ising model.

```bash
python scripts/basic_evolution.py
```

## Notebooks

The notebooks provide interactive tutorials:

1. **01_introduction.ipynb**: Basic concepts and first simulation
2. **02_custom_models.ipynb**: Creating custom quantum models
3. **03_analysis.ipynb**: Analyzing simulation results
4. **04_advanced_optimization.ipynb**: Advanced optimization techniques

## Running Notebooks

```bash
pip install -e ".[notebooks]"
jupyter notebook
```

## Contributing Examples

If you'd like to contribute an example:

1. Ensure your code is well-documented
2. Include expected output or figures
3. Keep examples focused on a single concept
4. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
