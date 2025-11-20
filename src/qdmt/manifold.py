from abc import ABC, abstractmethod
from ncon import ncon
import numpy as np
from qdmt.uniform_mps import UniformMps

class AbstractManifold(ABC):
    @abstractmethod
    def project(self, W: np.ndarray, D: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def retract(self, W: np.ndarray, X: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def transport(self, Y: np.ndarray, W: np.ndarray, X: np.ndarray, alpha: np.ndarray, W_new: np.ndarray) -> np.ndarray:
        pass


def euclidian_metric(X: np.ndarray, Y: np.ndarray):
    return np.real(np.trace(X.conj().T @ Y))

class Grassmann(AbstractManifold):
    def project(self, W: np.ndarray, D: np.ndarray) -> np.ndarray:
        if W.ndim == 2 and D.ndim == 2:
            return D - W @ W.conj().T @ D
        elif W.ndim == 3 and D.ndim == 3:
            return D - ncon((W, np.conj(W), D), ((-1, -2, 3), (1, 2, 3), (1, 2, -3)))
        else:
            raise ValueError("Error: Expected W and D to be either rank 3 (uMPS) or rank 2 (isometry) tensors!")
        
    def retract(self, W, X, alpha):

        # print(" before retraction")
        UniformMps(W).is_isometry()
        H = W.conj().T @ X
        h_norm = np.linalg.norm(H, ord="fro")
        # print(f"‖W†X‖ = {h_norm:.10e}  (should be ≈ 0 if X is horizontal)")

        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        c_s_alpha = np.diag(np.cos(s * alpha))
        s_s_alpha = np.diag(np.sin(s * alpha))

        W_prime = W @ (Vh.conj().T @ c_s_alpha @ Vh) + (U @ s_s_alpha @ Vh)

        s_s_alpha_s = np.diag(np.sin(s * alpha) * s)
        c_s_alpha_s = np.diag(np.cos(s * alpha) * s)
        X_prime = -W @ (Vh.conj().T @ s_s_alpha_s @ Vh) + (U @ c_s_alpha_s @ Vh)


        # ---- minimal polar polish (gentle, removes drift) ----
        # Compute S = W'† W' (Hermitian ≈ I) and apply S^{-1/2} via eig.
        S = W_prime.conj().T @ W_prime
        lam, Ue = np.linalg.eigh(S)
        lam = np.clip(lam, 1e-15, None)
        S_inv_sqrt = Ue @ np.diag(1.0 / np.sqrt(lam)) @ Ue.conj().T
        W_prime = W_prime @ S_inv_sqrt
        # keep tangent horizontal at new point (safe, 1 line)
        X_prime = X_prime - W_prime @ (W_prime.conj().T @ X_prime)
        # ------------------------------------------------------

        # print(" after retraction")
        # UniformMps(W_prime).is_isometry()

        return W_prime, X_prime
    
    def transport(self, Y, W, X, alpha, W_new):
        U, s, Vh = np.linalg.svd(X, full_matrices=False)

        UdY = U.conj().T @ Y
        WVh_conj = W @ Vh.conj().T

        cos_S_alpha = np.diag(np.cos(s * alpha))
        sin_S_alpha = np.diag(np.sin(s * alpha))

        Y_transported = Y + U @ ((cos_S_alpha - np.eye(len(s))) @ UdY) - WVh_conj @ (sin_S_alpha @ UdY)
        
        Y_new = self.project(W_new, Y_transported)
        
        return Y_new
        

class Steifel(AbstractManifold):
    def project(self, W: np.ndarray, D: np.ndarray) -> np.ndarray:
        if W.ndim == 2 and D.ndim == 2:
            return D - 0.5 * W @ (np.conj(W).T @ D + np.conj(D).T @ W)
        elif W.ndim == 3 and D.ndim == 3:
            raise NotImplementedError("Error: Not implemented Steifel projection on uMPS shape `(D, d, D)`")
        else:
            raise ValueError("Error: Expected W and D to be either rank 3 (uMPS) or rank 2 (isometry) tensors!")

    def retract(self, W, X, alpha):
        raise NotImplementedError()
    
    def transport(self, Y, W, X, alpha, W_new):
        raise NotImplementedError()