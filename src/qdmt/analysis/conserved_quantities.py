import numpy as np
from ncon import ncon

from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrix import TransferMatrix
from qdmt.model import AbstractModel

def compute_energy(A: UniformMps, model: AbstractModel) -> np.float64:
    # r = RightFixedPoint.from_mps(A)
    # print(A.tensor)
    r = TransferMatrix.new(A, A).right_fixed_point()
    # print(model.H)


    # h=np.eye(4).reshape(2, 2, 2, 2)
    h=model.H
    # print(np.trace(r.tensor))
    tensors = [A.tensor, A.tensor, h,  A.tensor.conj(), A.tensor.conj(), r.tensor]
    indices = [[1, 2, 3], [3, 4, 5], [2, 4, 6, 7], [1, 6, 8], [8, 7, 9], [5, 9]]
    # print(ncon(tensors, indices))
    return np.real(ncon(tensors, indices))
    # return np.trace(r.tensor)

# def compute_normalization(A: UniformMps):
#     r = TransferMatrix.new(A, A).right_fixed_point()
#     print(r)
#     # return A.normalization()

def compute_norm(A: UniformMps) -> np.float64:
    # r = RightFixedPoint.from_mps(A)
    # print(A.tensor)
    r = TransferMatrix.new(A, A).right_fixed_point()
    # print(model.H)


    h=np.eye(4).reshape(2, 2, 2, 2)
    # h=model.H
    # print(f"trace={np.abs(np.trace(r.tensor)):0.8e}")
    tensors = [A.tensor, A.tensor, h,  A.tensor.conj(), A.tensor.conj(), r.tensor]
    indices = [[1, 2, 3], [3, 4, 5], [2, 4, 6, 7], [1, 6, 8], [8, 7, 9], [5, 9]]
    # print(ncon(tensors, indices))
    return np.real(ncon(tensors, indices))
    # return np.trace(r.tensor)    

