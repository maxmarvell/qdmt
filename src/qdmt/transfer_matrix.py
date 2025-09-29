from abc import ABC, abstractmethod
import numpy as np
from numpy.linalg import LinAlgError
from ncon import ncon
from typing import Optional, Sequence
from scipy.sparse.linalg import LinearOperator, eigs, gmres
from dataclasses import dataclass
from functools import partial
import warnings

from qdmt.uniform_mps import UniformMps

class AbstractTransferMatrix(ABC):

    _derivative: Optional[np.ndarray]

    def __init__(self, array: np.ndarray, *, tape: Sequence["AbstractTransferMatrix"] = None):
        if not isinstance(array, np.ndarray):
            raise TypeError("E must be a numpy array.")
        if len(array.shape) % 2 != 0:
            raise ValueError("Balanced number of input and output legs required.")
        
        n = len(array.shape) // 2
        if array.shape[:n] != array.shape[-n:]:
            raise ValueError("Dimensions of input and output legs should be the same.")
        
        self._array = array
        self._n = n
        self._derivative = None
        self.tape = list(tape) if tape is not None else []

    @property
    def array(self) -> np.ndarray:
        return self._array
    
    @property
    def n(self) -> int:
        return self._n
    
    @abstractmethod
    def identity_like(self) -> "AbstractTransferMatrix":
        """blah"""

    @abstractmethod
    def derivative(self) -> np.ndarray:
        """Return d/dθ tensor (same shape as `tensor`)."""

    @abstractmethod
    def __matmul__(self, other) -> "AbstractTransferMatrix":
        pass

    @abstractmethod
    def _matvec(self, other) -> "AbstractTransferMatrix":
        pass

    def to_matrix(self) -> np.ndarray:
        N = int(np.prod(self.array.shape[:self.n]))
        return self.array.reshape((N, N))

    def linop(self) -> LinearOperator:
        return LinearOperator((self._n, self._n), matvec=self._matvec, dtype=self.array.dtype)
    
    def __pow__(self, n: int) -> "AbstractTransferMatrix":

        if n == 0:
            return self.identity_like()

        result = None
        power = self
        while n > 0:
            if n % 2 == 1:
                if result is None:
                    result = power
                else:
                    result = result @ power
            if n == 1:
                return result
            power = power @ power
            n //= 2
        return result

@dataclass
class RightFixedPoint():
    tensor: np.ndarray
    A: np.ndarray

    def derivative(self, v: np.ndarray) -> np.ndarray:
        _, nb = self.tensor.shape
        L = LinearOperator((nb ** 2, nb ** 2), matvec=partial(self._pseudoinverse_operator))
        Rh: np.ndarray = gmres(L, v.reshape(nb ** 2))[0]
        Rh = Rh.reshape(nb, nb)
        return ncon((Rh, self.E.B.array, self.array), ((-1, 1), (1, -2, 2), (2, -3)))
    
    def _pseudoinverse_operator(self, v: np.ndarray):
        d, _ = self.array.shape
        v = v.reshape(d, d)

        # transfermatrix contribution
        transfer = ncon((v, self.E.array), ((1, 2), (2, 1, -2, -1)))

        # fixed point contribution
        fixed = np.trace(v @ self.array) * np.eye(d)

        # sum these with the contribution of the identity
        return v - transfer + fixed


class TransferMatrix(AbstractTransferMatrix):

    def __init__(self, array: np.ndarray, *, tape: Sequence["TransferMatrix"] = None):
        super().__init__(array, tape=tape)

        # dimensions
        self.Da, self.Db = array.shape[:self.n]

        self._A: Optional[UniformMps] = None
        self._B: Optional[UniformMps] = None

    @property
    def A(self) -> Optional[UniformMps]:
        return self._A

    @property
    def B(self) -> Optional[UniformMps]:
        return self._B
    
    @classmethod
    def new(cls, A: UniformMps, B: UniformMps):
        E = ncon((A.tensor, B.tensor.conj()), ((-1, 1, -3), (-2, 1, -4)))
        obj = cls(E)
        obj._A, obj._B = A, B
        return obj
    
    def _compute_derivative(self) -> np.ndarray:
        if self.A == None or self.B == None:
            raise ValueError("Can only call _compute_derivative for non-composite transfer matrix object.")
        Ib = np.eye(self.Db)
        return ncon([self.A.tensor, Ib, Ib], ((-1, -6, -3), (-2, -5), (-7, -4)))
    
    def _matvec(self, v: np.ndarray):
        if self.A != None and self.B != None:
            return ncon([self.A.tensor, self.B.tensor, v], [[-1, 3, 1], [-2, 3, 2], [1, 2]])
        return ncon([self.array, v], [[-1, -2, 1, 2], [1, 2]])
    
    def _rmatvec(self, v):
        if self.A != None and self.B != None:
            return ncon([v, self.A.tensor, self.B.tensor], [[1, 2], [-1, 3, 2], [-2, 3, 1]])
        return ncon([self.array, v], [[-1, -2, 1, 2], [1, 2]])

    def __matmul__(self, other):

        if isinstance(other, TransferMatrix):
            tensor = ncon((self.array, other.array), ((-1, -2, 1, 2), (1, 2, -3, -4)))
            return TransferMatrix(tensor, [self, other])
        
        if isinstance(other, np.ndarray):
            if other.ndim == 2:
                return self._matvec(other)
            else:
                raise ValueError("Unsupported operand shape for multiplication.")
            
        raise NotImplementedError()

    def __rmatmul__(self, other):
        if isinstance(other, np.ndarray) and other.ndim == 2:
            return self._rmatvec(other)
        raise NotImplementedError()
    
    def derivative(self):

        if self._derivative is not None:
            return self._derivative
        
        # if no parent tensors simply compute derivative and return
        elif len(self.tape) == 0:
            self._derivative = self._compute_derivative()
            return self._derivative
        
        elif len(self.tape) == 2:
            L = self.tape[0]
            R = self.tape[1]

            indices = [[-1, -2, 1, 2, -5, -6, -7], [1, 2, -3, -4]]
            LRdL = ncon([L.derivative(), R.array], indices)

            indices = [[-1, -2, 1, 2], [1, 2, -3, -4, -5, -6, -7]]
            LRdR = ncon([L.array, R.derivative()], indices)

            self._derivative = LRdL + LRdR
            return self._derivative
        
        raise NotImplementedError
    
    def identity_like(self):
        I = np.eye(self.n).reshape(*self.array.shape)
        return TransferMatrix(I)
    
    def right_fixed_point(self):
        if self.A is None or self.B is None:
            raise ValueError("Need A and B to define fixed point.")
        
        if self.A is not self.B:
            warnings.warn("A and B differ: right_fixed_point is not guaranteed.")

        M = self.to_matrix()
        eig, r = eigs(M, k=1, which='LM')

        if np.allclose(eig, [1.]):
            r = r.reshape((self.Da, self.Db))
            r /= (np.trace(r) / np.abs(np.trace(r)))
            r = (r + np.conj(r).T) / 2
            r *= np.sign(np.trace(r))
            return RightFixedPoint(r, self.A)

        raise LinAlgError("Transfer matrix has no eigenvalue 1: right fixed point did not converge.")


class FirstOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    def __init__(self, array: np.ndarray, *, tape: Sequence["FirstOrderTrotterizedTransferMatrix"] = None):
        super().__init__(array, tape=tape)
        self.Da, self.d, self.Db = array.shape[:self.n]
        self._A: Optional[UniformMps] = None
        self._B: Optional[UniformMps] = None
        self._U1: Optional[np.ndarray] = None
        self._U2: Optional[np.ndarray] = None

    @property
    def A(self) -> Optional[UniformMps]:
        return self._A

    @property
    def B(self) -> Optional[UniformMps]:
        return self._B
    
    @property
    def U1(self) -> Optional[np.ndarray]:
        return self._U1

    @property
    def U2(self) -> Optional[np.ndarray]:
        return self._U2

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps, U1: np.ndarray, U2: np.ndarray):

        tensors = [B.conj, B.conj, U2, U1, A.tensor, A.tensor]
        indices = [(-3, 1, 2), (2, 3, -6), (-2, 4, 1, 3), (5, 6, 4, -5), (-1, 5, 7), (7, 6, -4)]
        E = ncon(tensors, indices)

        # create object
        obj = FirstOrderTrotterizedTransferMatrix(E)
        obj._A, obj._B, obj._U1, obj._U2 = A, B, U1, U2
        return obj
    
    def __matmul__(self, other):
        if isinstance(other, FirstOrderTrotterizedTransferMatrix):
            indices = [[-1, -2, -3, 1, 2, 3], [1, 2, 3, -4, -5, -6]]
            tensor = ncon([self.array, other.array], indices, order=[1, 3, 2])
            return FirstOrderTrotterizedTransferMatrix(tensor, [self, other])

        raise NotImplementedError()
    
    def derivative(self):

        if self._derivative is not None:
            return self._derivative

        if len(self.tape) == 0:

            self._derivative = self._compute_derivative()

            return self._derivative
        
        if len(self.tape) == 2:

            L = self.tape[0]
            R = self.tape[1]

            tensors = [L.derivative(), R.array]
            indices = [[-1, -2, -3, 1, 2, 3, -7, -8, -9], [1, 2, 3, -4, -5, -6]]
            LRdL = ncon(tensors, indices)

            tensors = [L.array, R.derivative()]
            indices = [[-1, -2, -3, 1, 2, 3], [1, 2, 3, -4, -5, -6, -7, -8, -9]]
            LRdR = ncon(tensors, indices)

            self.D = LRdL + LRdR

            return self.D
        
        raise NotImplementedError()

    def _compute_derivative(self) -> np.ndarray:

        if self.A == None or self.B == None or self.U1 == None or self.U2 == None:
            raise ValueError("Can only call _compute_derivative for non-composite transfer matrix object.")
        
        Ib = np.eye(self.Db)
        tensors = [Ib, self.B.tensor.conj(), self.U2, self.U1, self.A.tensor, self.A.tensor]
        indices = [[-3, -7], [-9, 1, -6], [-2, 2, -8, 1], [3, 4, 2, -5], [-1, 3, 5], [5, 4, -4]]
        D1 = ncon(tensors, indices)

        tensors = [self.B.tensor.conj(), Ib, self.U2, self.U1, self.A.tensor, self.A.tensor]
        indices = [[-3, 1, -7], [-9, -6], [-2, 2, 1, -8], [3, 4, 2, -5], [-1, 3, 5], [5, 4, -4]]
        D2 = ncon(tensors, indices)

        self.D = D1 + D2

        return self.D
            
class SecondOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    _A: Optional[UniformMps]
    _B: Optional[UniformMps]
    _U1: Optional[np.ndarray]
    _U2: Optional[np.ndarray]

    def __init__(self, array, *, tape: Sequence["SecondOrderTrotterizedTransferMatrix"] = None):
        super().__init__(array, tape=tape)
        self.Da, self.d, _, self.Db = array.shape[:self.n]
        self._A, self._B, self._U1, self._U2 = None, None, None, None

    @property
    def A(self) -> Optional[UniformMps]:
        return self._A

    @property
    def B(self) -> Optional[UniformMps]:
        return self._B
    
    @property
    def U1(self) -> Optional[np.ndarray]:
        return self._U1

    @property
    def U2(self) -> Optional[np.ndarray]:
        return self._U2

    @classmethod
    def new(cls, A: UniformMps, B: UniformMps, U1: np.ndarray, U2: np.ndarray):

        tensors = [B.tensor.conj(), B.tesnor.conj(), U1, U2, U1, A.tensor, A.tensor]
        indices = [[-4, 1, 2], [2, 3, -8], [-3, 4, 1, 3], [-2, 5, 4, -7], [6, 7, 5, -6], [-1, 6, 8], [8, 7, -5]]
        E = ncon(tensors, indices)

        obj = SecondOrderTrotterizedTransferMatrix(E)
        obj.A, obj.B, obj.U1, obj.U2 = A, B, U1, U2
        return obj

    def __matmul__(self, other):
        if isinstance(other, SecondOrderTrotterizedTransferMatrix):
            indices = [[-1, -2, -3, -4, 1, 2, 3, 4], [1, 2, 3, 4, -5, -6, -7, -8]]
            tensor = ncon([self.array, other.array], indices, order=[1, 4, 2, 3])
            return SecondOrderTrotterizedTransferMatrix(tensor, [self, other])
            
        raise NotImplementedError()

    def _compute_derivative(self) -> None:

        if self.A == None or self.B == None or self.U1 == None or self.U2 == None:
            raise ValueError("Can only call _compute_derivative for non-composite transfer matrix object.")
        
        A, B, U1, U2 = self.A.tensor, self.B.tensor, self.U1, self.U2

        tensors = [A, A, U1, U2, U1]
        indices = [[-1, 1, 2], [2, 3, -5], [1, 3, 4, -6], [-2, 4, 5, -7], [-3, 5, -4, -8]]
        order = [2, 1, 3, 4, 5]
        res = ncon(tensors, indices, order=order)

        tensors = [np.eye(self.Db), B.conj(), res]
        indices = [[-4, -9], [-11, 1, -8], [-1, -2, -3, -10, -5, -6, -7, 1]]
        deriv_1 = ncon(tensors, indices)

        tensors = [B.conj(), res, np.eye(self.Db)]
        indices = [[-4, 1, -9], [-1, -2, -3, 1, -5, -6, -7, -10], [-11, -8]]
        deriv_2 = ncon(tensors, indices)

        self._derivative = deriv_1 + deriv_2
        

    def derivative(self):
        
        if self._derivative is not None:
            return self._derivative
        
        # if no parent tensors simply compute derivative and return
        if len(self.tape) == 0:
            self._compute_derivative()
            return self._derivative
        
        if len(self.tape) == 2:
            L = self.tape[0]
            R = self.tape[1]

            tensors = [L.derivative(), R.array]
            indices = [
                [-1, -2, -3, -4, 1, 2, 3, 4, -9, -10, -11], 
                [1, 2, 3, 4, -5, -6, -7, -8]
            ]
            D1 = ncon(tensors, indices, order=[1, 4, 2, 3])
            
            tensors = [L.array, R.derivative()]
            indices = [
                [-1, -2, -3, -4, 1, 2, 3, 4], 
                [1, 2, 3, 4, -5, -6, -7, -8, -9, -10, -11]
            ]
            D2 = ncon(tensors, indices, order=[1, 4, 2, 3])

            self._derivative = D1 + D2
            return self._derivative
        
        raise NotImplementedError()

if __name__ == "__main__":
    
    A = UniformMps.random(4, 2)
    E = TransferMatrix.new(A, A)
    E.right_fixed_point()