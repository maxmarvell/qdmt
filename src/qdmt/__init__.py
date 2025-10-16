"""
QDMT: Quantum Density Matrix Truncation

A Python package for efficient quantum time evolution using Matrix Product State (MPS)
representations with density matrix truncation methods.
"""

__version__ = "0.1.0"

# Core MPS functionality
from qdmt.uniform_mps import UniformMps
from qdmt.isometry import Isometry

# Physical models
from qdmt.model import (
    AbstractModel,
    TransverseFieldIsing,
    HeisenbergXXZ,
    Pauli,
)

# Cost functions
from qdmt.cost import (
    AbstractCostFunction,
    HilbertSchmidt,
    EvolvedHilbertSchmidt,
    NaiveEvolvedHilbertSchmidt,
)

# Optimization
from qdmt.optimisation import ConjugateGradient
from qdmt.manifold import Grassmann

# Transfer matrices and fixed points
from qdmt.transfer_matrix import (
    TransferMatrix,
    RightFixedPoint,
    FirstOrderTrotterizedTransferMatrix,
    SecondOrderTrotterizedTransferMatrix,
)

# Time evolution
from qdmt.evolve import evolve

__all__ = [
    # Version
    "__version__",

    # MPS
    "UniformMps",
    "Isometry",

    # Models
    "AbstractModel",
    "TransverseFieldIsing",
    "HeisenbergXXZ",
    "Pauli",

    # Cost functions
    "AbstractCostFunction",
    "HilbertSchmidt",
    "EvolvedHilbertSchmidt",
    "NaiveEvolvedHilbertSchmidt",

    # Optimization
    "ConjugateGradient",
    "Grassmann",

    # Transfer matrices
    "TransferMatrix",
    "RightFixedPoint",
    "FirstOrderTrotterizedTransferMatrix",
    "SecondOrderTrotterizedTransferMatrix",

    # Evolution
    "evolve",
]
