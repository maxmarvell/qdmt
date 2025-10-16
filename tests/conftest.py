"""
Pytest configuration and shared fixtures for QDMT tests.
"""

import pytest
import numpy as np
from qdmt.uniform_mps import UniformMps
from qdmt.model import TransverseFieldIsing


@pytest.fixture
def A():
    """Create a random uniform MPS."""
    return UniformMps.random(D=5, d=2)


@pytest.fixture
def B():
    """Create a random uniform MPS."""
    return UniformMps.random(D=8, d=2)


@pytest.fixture
def tfim_model():
    """Create a Transverse Field Ising model."""
    return TransverseFieldIsing(g=1.0, delta_t=0.01)


@pytest.fixture(params=[4, 6, 8, 10])
def L(request):
    """patch size parameter for tests."""
    return request.param


@pytest.fixture
def small_system_params():
    """Parameters for small test systems."""
    return {
        'D': 4,
        'd': 2,
        'L': 10,
        'delta_t': 0.01,
        'max_t': 0.1,
        'tol': 1e-6
    }


# Configure numpy for reproducible tests
np.random.seed(42)
