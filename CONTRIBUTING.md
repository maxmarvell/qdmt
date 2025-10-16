# Contributing to QDMT

Thank you for your interest in contributing to QDMT! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/qdmt.git
   cd qdmt
   ```

3. Install in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, following our coding standards

3. Add tests for new functionality

4. Run the test suite:
   ```bash
   pytest
   ```

5. Check code quality:
   ```bash
   ruff check src/ tests/
   mypy src/
   ```

6. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request on GitHub

## Coding Standards

### Python Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use descriptive variable names
- Write docstrings for all public functions, classes, and modules

### Docstring Format

Use NumPy-style docstrings:

```python
def function(arg1: int, arg2: str) -> bool:
    """
    Brief description.

    Longer description if needed.

    Parameters
    ----------
    arg1 : int
        Description of arg1
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value

    Examples
    --------
    >>> function(1, "test")
    True
    """
    pass
```

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Use pytest fixtures for common test setups
- Name test files as `test_*.py`
- Name test functions as `test_*`

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

Longer explanation if necessary. Wrap at 72 characters.
Explain what changed and why, not how.

- Bullet points are okay
- Use present tense: "Add feature" not "Added feature"
```

## Types of Contributions

### Bug Reports

- Use the issue tracker
- Include a minimal reproducible example
- Provide system information (OS, Python version, package versions)
- Describe expected vs actual behavior

### Feature Requests

- Open an issue to discuss before implementing
- Explain the use case and benefit
- Consider if it fits the project scope

### Documentation

- Fix typos, clarify explanations
- Add examples and tutorials
- Improve API documentation

### Code Contributions

Priority areas:
- Performance optimizations
- Additional physical models
- Analysis tools
- Test coverage improvements

## Pull Request Process

1. **Before submitting**:
   - Ensure tests pass
   - Update documentation
   - Add entry to CHANGELOG.md
   - Run code quality checks

2. **PR description should include**:
   - What changed and why
   - Related issue numbers
   - Any breaking changes
   - Testing performed

3. **Review process**:
   - Maintainers will review your PR
   - Address feedback and questions
   - Once approved, we'll merge your contribution

4. **After merging**:
   - Your contribution will be included in the next release
   - You'll be added to the contributors list

## Running Tests Locally

### Full test suite
```bash
pytest
```

### With coverage
```bash
pytest --cov=qdmt --cov-report=html
```

### Specific test file
```bash
pytest tests/test_mps.py
```

### Benchmarks
```bash
pytest benchmarks/ --benchmark-only
```

## Building Documentation

```bash
cd docs
make html
```

View the built docs at `docs/_build/html/index.html`.

## Questions?

- Open an issue for bugs or feature requests
- Use GitHub Discussions for questions and general discussion
- Check existing issues and docs before asking

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

All contributors will be acknowledged in the project README and release notes.

Thank you for contributing to QDMT!
