# Tests

This directory contains the test suite for QDMT.

## Structure

- `unit/`: Fast, isolated unit tests for individual components
- `integration/`: Tests for component interactions
- `helpers/`: Shared test utilities and helper functions
- `conftest.py`: Pytest configuration and fixtures

## Running Tests

### All tests
```bash
pytest
```

### Unit tests only
```bash
pytest tests/unit/
```

### Integration tests only
```bash
pytest tests/integration/
```

### Specific test file
```bash
pytest tests/unit/test_mps.py
```

### With coverage
```bash
pytest --cov=qdmt --cov-report=html
```

### Verbose output
```bash
pytest -v
```

### Run only fast tests (skip slow ones)
```bash
pytest -m "not slow"
```

## Writing Tests

### Test Organization

- Unit tests: Test individual functions/classes in isolation
- Integration tests: Test multiple components working together
- Use fixtures from `conftest.py` for common test setups

### Example Test

```python
import pytest
from qdmt.uniform_mps import UniformMps

def test_mps_creation():
    """Test that MPS creation works correctly."""
    D, d = 5, 2
    A = UniformMps.random(D=D, d=d)
    assert A.tensor.shape == (D, d, D)

@pytest.mark.slow
def test_expensive_computation():
    """Mark slow tests to skip them during fast test runs."""
    # Long-running test...
    pass
```

### Using Fixtures

```python
def test_with_fixture(random_mps):
    """Use the random_mps fixture from conftest.py."""
    assert random_mps.is_normalized()
```

## Continuous Integration

Tests are automatically run on:
- Every push to main/develop branches
- Every pull request
- Multiple Python versions (3.11, 3.12)
- Multiple operating systems (Ubuntu, macOS)

See `.github/workflows/tests.yml` for CI configuration.
