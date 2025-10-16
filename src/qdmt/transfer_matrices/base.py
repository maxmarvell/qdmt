import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres, eigs
from functools import partial
from ncon import ncon
from abc import ABC, abstractmethod
from typing import Tuple
from qdmt.uniform_mps import UniformMps
from numpy.linalg import LinAlgError
import warnings
from numpy.typing import NDArray

class RightFixedPoint():
    def __init__(self, tensor: np.ndarray, A: np.ndarray):
        self.tensor, self.A = tensor, A
        self.D = tensor.shape[0]

    def derivative(self, v: np.ndarray) -> np.ndarray:
        L = LinearOperator((self.D ** 2, self.D ** 2), matvec=partial(self._pseudoinverse_operator))
        Rh: np.ndarray = gmres(L, v.reshape(self.D ** 2))[0]
        Rh = Rh.reshape(self.D, self.D)
        return ncon((Rh, self.A, self.tensor), ((-1, 1), (1, -2, 2), (2, -3)))
    
    def _pseudoinverse_operator(self, v: np.ndarray):
        # reshape as a matrix
        v = v.reshape(self.D, self.D)

        # get the transfermatrix contribution
        transfer = ncon([v, self.A, self.A.conj()], [[1, 2], [2, 3, -2], [1, 3, -1]])

        # fixed point contribution
        fixed = np.trace(v @ self.tensor) * np.eye(self.D)

        # sum these with the contribution of the identity
        return v - transfer + fixed

class AbstractTransferMatrix(ABC):

    def __init__(self, n: int, dtype: np.dtype):
        self.n, self.dtype = n, dtype

    @abstractmethod
    def _matvec(self, other: NDArray[np.complex128]) -> NDArray[np.complex128]:
        pass

    @abstractmethod
    def _rmatvec(self, other: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Transposes the vector and performs a right matrix-vector product."""

    @abstractmethod
    def _left_at(self, *index: Tuple[int]):
        """Returns the a right vector."""

    @abstractmethod
    def _deriv_right_at(self, index: Tuple[int]):
        pass

    @property
    @abstractmethod
    def matrix(self):
        pass

    def linop(self) -> LinearOperator:
        return LinearOperator((self.n, self.n), matvec=self._matvec, dtype=self.dtype)
    

class TransferMatrix(AbstractTransferMatrix):

    def __init__(self, A: UniformMps, B: UniformMps):
        self.Da, self.d, self.Db = A.D, A.d, B.D
        self.A, self.B = A, B

        n = self.Da * self.Db
        super().__init__(n, A.tensor.dtype)

    def _rmatvec(self, other: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Performs right matrix-vector product.
        """

        if isinstance(other, np.ndarray):

            if other.ndim == 1 and other.size == self.Da * self.Db:
                other = other.reshape(self.Da, self.Db)
            elif other.ndim != 2:
                raise ValueError("_rmatvec only supported for vectors of dim (Da, Db).")
            
            tmp = np.einsum('acd,ab->bcd', self.A.tensor, other)
            return np.einsum('bcd,bce->de', tmp, self.B.tensor.conj())
            
        raise NotImplementedError("_matvec only implemented for numpy arrays.")

    def _matvec(self, other: NDArray[np.complex128]) -> NDArray[np.complex128]:

        if isinstance(other, np.ndarray):

            if other.ndim == 1 and other.size == self.Da * self.Db:
                other = other.reshape(self.Da, self.Db)
            elif other.ndim != 2:
                raise ValueError("_matvec only supported for vectors of dim (Da, Db).")
            
            tmp = np.einsum('acd,de->ace', self.A.tensor, other)
            return np.einsum('ace,bce->ab', tmp, self.B.tensor.conj())
        
        raise NotImplementedError("_matvec only implemented for numpy arrays.")
    
    def right_fixed_point(self):        
        if self.A is not self.B:
            warnings.warn("A and B differ: right_fixed_point is not guaranteed.")

        eig, r = eigs(self.matrix, k=1, which='LM')

        if np.allclose(eig, [1.]):
            r = r.reshape((self.Da, self.Db))
            r /= (np.trace(r) / np.abs(np.trace(r)))
            r = (r + np.conj(r).T) / 2
            r *= np.sign(np.trace(r))
            return RightFixedPoint(r, self.A)

        raise LinAlgError("Transfer matrix has no eigenvalue 1: right fixed point did not converge.")
    
    def _left_at(self, *index: Tuple[int]):
        i, j = index
        return np.einsum('ba,bc->ac', self.A.tensor[i,:,:], self.B.tensor.conj()[j,:,:])
    
    def _deriv_right_at(self, v: np.ndarray, index: Tuple[int]) -> np.ndarray:
        i, j = index
        tmp = np.einsum('ac,ab->cb', v, A.tensor[:,:,i])
        return np.einsum('cb,e->bce', tmp, np.eye(D)[:,j])
    
    @property
    def matrix(self):
        M: np.ndarray = np.einsum('abc,dbe->adce', self.A.tensor, self.B.tensor.conj())
        return M.reshape((self.Da*self.Db, self.Da*self.Db))
    


if __name__ == "__main__":

    # construct random initial uMPS
    D, d = 8, 2
    A = UniformMps.random(D, d)

    # construct arbitrary vector
    v = np.random.rand(D, D) + 1j * np.random.rand(D, D)

    import opt_einsum as oe

    # scaling of computing _rmatvec()
    tensors = (v, A.tensor, A.tensor.conj())
    path = oe.contract_path('ab,acd,bce->de', *tensors)
    print(path)

    # scaling of computing _deriv_right_at()
    tensors = [v, A.tensor[:,:,0], np.eye(D)[:,0]]
    path = oe.contract_path('ab,ac,e->bce', *tensors)
    print(path)

    E = TransferMatrix(A, A)

    print(E.matrix.shape)
