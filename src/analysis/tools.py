import numpy as np
from ncon import ncon
from qdmt.uniform_mps import UniformMps
from qdmt.cost import HilbertSchmidt
from qdmt.transfer_matrix import TransferMatrix
from qdmt.fixed_point import RightFixedPoint

def compute_second_Reyni(A: UniformMps, L: int):
    C = HilbertSchmidt(A, L)
    return -np.log(C.costAA)

def log_negativity(A: UniformMps, L: int):
    r = RightFixedPoint.from_mps(A)
    E = TransferMatrix(A, A)
    E = E.__pow__(L)
    norm = ncon([E, r], [[1, 2, 3, 4], [4, 3]])