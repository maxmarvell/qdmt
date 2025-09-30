from ncon import ncon
import numpy as np
from scipy.sparse.linalg import eigs
from qdmt.isometry import Isometry
from typing import Optional

class UniformMps(Isometry):

    # uMPS dimensions fully define tensor A (D, d, D)
    d: int
    D: int

    def __init__(self, A: np.ndarray, *, tol: float = 1e-10):
        if A.ndim == 3:
            _D, d, D = A.shape
            if _D != D:
                raise ValueError("Bond dimensions must match: (D, d, D)")
        elif A.ndim == 2:
            dD, D = A.shape
            if dD % D != 0:
                raise ValueError("Isometry shape must be like: (D*d, D)")
            d = dD // D
        else:
            raise ValueError("A must have shape: (D, d, D) or (D*d, D)")
        
        V = np.ascontiguousarray(A.reshape(d * D, D, order="C"))
        super().__init__(V=V, tol=tol)

        # cached property
        self._tensor: Optional[np.ndarray] = None
        
        self.d = d
        self.D = D

    @property
    def tensor(self) -> np.ndarray:
        if self._tensor is None:
            d, D = self.d, self.D
            order = "C" if self.V.flags.c_contiguous else ("F" if self.V.flags.f_contiguous else None)
            
            if order is None:
                self.V = np.ascontiguousarray(self.V)
                order = "C"
            self._tensor = self.V.reshape(D, d, D, order=order)

        return self._tensor
    
    def __setattr__(self, name, value):
        if name == "V" and "_tensor" in self.__dict__:
            object.__setattr__(self, "_tensor", None)
        super().__setattr__(name, value)
    
    @classmethod
    def random(cls, D: int, d: int, *, seed: int | None = None, tol: float = 1e-10) -> "UniformMps":
        m, n = D * d, D
        return super().random(m, n, seed=seed, tol=tol)
    
    def inner_product(self, other: "UniformMps") -> np.complex128:
        E = ncon((self.tensor, other.tensor.conj()), ((-1, 1, -3), (-2, 1, -4)))
        r = eigs(E.reshape(self.D*other.D, self.D*other.D), k=1, which='LM', return_eigenvectors=False)
        return r[0]
    
    def fidelity(self, other: "UniformMps") -> np.float64:
        overlap = self.inner_product(other)
        return np.real(overlap * overlap.conj())
    
    def correlation_length(self) -> np.float64:
        E = ncon((self.tensor, self.tensor.conj()), ((-1, 1, -3), (-2, 1, -4)))
        r = eigs(E.reshape(self.D**2, self.D**2), k=2, which='LM', return_eigenvectors=False)
        r_sorted = np.sort(np.abs(r))[::-1]
        return -1/np.log(np.abs(r_sorted[1])/np.abs(r_sorted[0]))
    
    def norm(self) -> np.float64:
        return self.fidelity(self)
    
    def to_mps_chain(self, L: int) -> np.ndarray:
        if L == 0:
            return np.eye(self.D)
        if L == 1:
            return self.tensor
        tensors = [self.tensor for _ in range(L)]
        indices = [[-i, -(i+1), i] if i == 1 else [i-1, -(i+1), -(i+2)] if i == L else [i-1, -(i+1), i] for i in range(1, L+1)]
        return ncon(tensors, indices)
 
if __name__ == "__main__":
    theta = phi = np.pi / 2

    D = 6
    psi = np.array([np.cos(theta/2), np.exp(phi*1j)*np.sin(theta/2)])

    A = np.zeros((D, 2, D), dtype=np.complex128)
    A[0, 0, 0] = np.cos(theta / 2)
    A[0, 1, 0] = np.exp(phi * 1j) * np.sin(theta / 2)

    uMPS = UniformMps(A)

    uMPS.tensor.conj()

    print(psi)

    mps = UniformMps(A)

    print(mps.is_isometry())