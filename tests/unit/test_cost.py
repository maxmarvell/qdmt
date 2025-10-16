from qdmt.cost import EvolvedHilbertSchmidt as Naive
from qdmt.cost_fns import TimeEvolvedHilbertSchmidt as Efficient
from qdmt.model import TransverseFieldIsing
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrices import TransferMatrix
import numpy as np

def test_time_evolved_purity(A, L):

    model = TransverseFieldIsing(0.1, 0.2)
    f = Efficient(A, model, L)
    _f = Naive(A, model, L)

    assert np.allclose(
        f._compute_second_trotterized_purity(),
        _f._compute_second_trotterized_purity()
    )

def test_varitional_purity(A: UniformMps, B: UniformMps, L: int):

    rB = TransferMatrix(B, B).right_fixed_point()

    model = TransverseFieldIsing(0.1, 0.2)
    f = Efficient(A, model, L)
    _f = Naive(A, model, L)

    assert np.allclose(
        f._compute_variational_purity(B, rB),
        _f._compute_trace_product_rhoB_rhoB(B, rB)
    )