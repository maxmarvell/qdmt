from qdmt.transfer_matrices import TransferMatrix
from qdmt.model import TransverseFieldIsing
from qdmt.uniform_mps import UniformMps
import numpy as np
import pytest

from tests.conftest import A, L

def test_matvec(A: UniformMps):

    E = TransferMatrix(A, A)
    v = np.random.rand(A.D, A.D) + 1j * np.random.rand(A.D, A.D)
    v_flat = v.reshape(-1)

    res = E._matvec(v)

    assert np.allclose(
        res.reshape(-1),
        E.matrix @ v_flat
    )

def test_rmatvec(A: UniformMps):

    E = TransferMatrix(A, A)
    v = np.random.rand(A.D, A.D) + 1j * np.random.rand(A.D, A.D)
    v_flat = v.reshape(-1)

    res = E._rmatvec(v)

    assert np.allclose(
        res.reshape(-1),
        v_flat @ E.matrix
    )

def test_left_at(A: UniformMps):
    """Check that """

    E = TransferMatrix(A, A)
    v = E._left_at(1, 2)
    M = E.matrix.reshape(E.Da, E.Db, E.Da, E.Db)
    
    assert np.allclose(v, M[1,2,:,:])




# def test_right_matvec(A: UniformMps):

#     E = TransferMatrix(A, A)

#     v = np.random.rand((A.D, A.D)) + 1j * np.random.rand()

#     assert np.allclose(
#         f._compute_second_trotterized_purity(),
#         _f._compute_second_trotterized_purity()
#     )
