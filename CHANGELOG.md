# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project restructuring for scientific Python best practices
- Full documentation structure with Sphinx support
- GitHub Actions CI/CD pipeline
- Pre-commit hooks for code quality
- Example scripts and notebooks
- Type hints and mypy configuration
- Ruff for linting and formatting

### Changed
- Renamed `utils_n/` to `utils/` for clarity
- Moved `src/analysis/` to `src/qdmt/analysis/` for proper package structure
- Updated all import paths to reflect new structure
- Enhanced `pyproject.toml` with modern Python packaging standards

### Fixed
- Import paths after restructuring

## [0.1.0] - 2024-10-13

### Added
- Initial release
- Uniform MPS (Matrix Product State) implementation
- Time evolution using Trotterization (1st and 2nd order)
- Transfer matrix methods for fixed point calculations
- Hilbert-Schmidt cost function with evolved states
- Riemannian optimization on Grassmann manifold
- Transverse Field Ising model
- Heisenberg XXZ model
- Command-line interface for simulations
- Analysis tools:
  - Magnetization calculations
  - Correlation length
  - Schmidt values
  - Loschmidt echo
  - Conserved quantities
- Benchmark suite for performance profiling
- Basic test coverage

### Known Issues
- Some tests may require updates after restructuring
- Documentation is work in progress

[Unreleased]: https://github.com/yourusername/qdmt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/qdmt/releases/tag/v0.1.0
