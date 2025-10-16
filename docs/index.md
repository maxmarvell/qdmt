# QDMT Documentation

Welcome to the documentation for **QDMT** (Quantum Density Matrix Truncation).

## Overview

QDMT is a Python package for efficient quantum time evolution using Matrix Product State (MPS) representations with density matrix truncation methods.

## Quick Links

- [Installation Guide](installation.md)
- [Quick Start Tutorial](tutorials/quickstart.md)
- [API Reference](api/index.md)
- [Theory Background](theory/index.md)
- [Examples](examples/index.md)

## Key Features

- Time evolution of quantum states using Trotterization
- Density matrix truncation with controlled bond dimension
- Optimization on Riemannian manifolds
- Built-in physical models (Ising, Heisenberg)
- Comprehensive analysis tools

## Getting Started

```python
from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing
from qdmt.evolve import evolve

# Initialize MPS
A0 = UniformMps.random(D=10, d=2)

# Define model
model = TransverseFieldIsing(g=1.0, delta_t=0.01)

# Evolve
times, states, cost, norm = evolve(
    A0=A0, D=10, L=100, model=model,
    delta_t=0.01, max_t=1.0,
    max_iter=100, tol=1e-6
)
```

## Table of Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
tutorials/index
examples/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
theory/index
```

```{toctree}
:maxdepth: 1
:caption: Development

contributing
changelog
```

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/qdmt/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/qdmt/discussions)

## Citation

If you use this software in your research, please cite:

```bibtex
@software{qdmt2024,
  title = {QDMT: Quantum Density Matrix Truncation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/qdmt}
}
```

## License

QDMT is released under the [MIT License](https://github.com/yourusername/qdmt/blob/main/LICENSE).
