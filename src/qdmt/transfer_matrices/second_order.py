from qdmt.transfer_matrices.base import AbstractTransferMatrix
from typing import Tuple
import numpy as np
from qdmt.uniform_mps import UniformMps

class SecondOrderTrotterizedTransferMatrix(AbstractTransferMatrix):

    def __init__(self, A: UniformMps, B: UniformMps, U1: np.ndarray, U2: np.ndarray):
        self.Da, self.d, self.Db = A.D, A.d, B.D
        self.A, self.B, self.U1, self.U2 = A, B, U1, U2

        n = self.Da * self.d ** 2 * self.Db
        super().__init__(n, U1.dtype)

    def _matvec(self, other: np.ndarray):

        if isinstance(other, np.ndarray):

            if other.ndim == 1 and other.size == self.Da * self.d ** 2 * self.Db:
                other = other.reshape(self.Da, self.d, self.d, self.Db)
            elif other.ndim != 4:
                raise ValueError("_matvec only supported for vectors of dim (Da, d, d, Da).")

            tmp1 = np.einsum('fgh,aef->ghae', self.A.tensor, self.A.tensor)
            tmp2 = np.einsum('ghae,egij->haij', tmp1, self.U1)
            tmp3 = np.einsum('haij,hjlp->ailp', tmp2, other)
            tmp4 = np.einsum('ailp,bikl->apbk', tmp3, self.U2)
            tmp5 = np.einsum('onp,dmo->npdm', self.B.tensor.conj(), self.B.tensor.conj())
            tmp6 = np.einsum('npdm,ckmn->pdck', tmp5, self.U1)
            return np.einsum('pdck,apbk->abcd', tmp6, tmp4)
        
        raise NotImplementedError("_matvec only implemented for numpy arrays.")

    def _rmatvec(self, other: np.ndarray):

        if isinstance(other, np.ndarray):

            if other.ndim == 1 and other.size == self.Da * self.d ** 2 * self.Db:
                other = other.reshape(self.Da, self.d, self.d, self.Db)
            elif other.ndim != 4:
                raise ValueError("_rmatvec only supported for vectors of dim (Da, d, d, Da).")

            tmp1 = np.einsum('fgh,aef->ghae', self.A.tensor, self.A.tensor)
            tmp2 = np.einsum('ghae,egij->haij', tmp1, self.U1)
            tmp3 = np.einsum('onp,dmo->npdm', self.B.tensor.conj(), self.B.tensor.conj())
            tmp4 = np.einsum('npdm,ckmn->pdck', tmp3, self.U1)
            tmp5 = np.einsum('pdck,abcd->pkab', tmp4, other)
            tmp6 = np.einsum('pkab,bikl->pail', tmp5, self.U2)
            return np.einsum('pail,haij->hjlp', tmp6, tmp2)
        
        raise NotImplementedError("_rmatvec only implemented for numpy arrays.")
    
    def _left_at(self, *index: Tuple[int]):
        i, j, k, l = index
        tmp1 = np.einsum('bcd,ab->cda', self.A.tensor, self.A.tensor[i,:,:])
        tmp2 = np.einsum('kjl,ik->jli', self.B.tensor.conj(), self.B.tensor.conj()[l,:,:])
        tmp3 = np.einsum('jli,gij->lg', tmp2, self.U1[k,:,:,:])
        tmp4 = np.einsum('cda,acef->def', tmp1, self.U1)
        tmp5 = np.einsum('lg,egh->leh', tmp3, self.U2[j,:,:,:])
        return np.einsum('leh,def->dfhl', tmp5, tmp4)
    
    def _deriv_right_at(self, v: np.ndarray, index: Tuple[int]):
        i, j, k, l = index

        D = np.empty_like(self.B.tensor)

        tmp1 = np.einsum('bhi,egh->bieg', self.U1[:,:,:,j], self.U2[:,:,:,k])
        tmp2 = np.einsum('fg,aef->gae', self.A.tensor[:,:,i], self.A.tensor)
        tmp3 = np.einsum('gae,bieg->abi', tmp2, tmp1)
        tmp4 = np.einsum('abi,abcd->icd', tmp3, v)
        tmp5 = np.einsum('icd,cijk->djk', tmp4, self.U1)

        # compute first contribution
        D += np.einsum('djk,lk->djl', tmp5, self.B.tensor.conj()[:,:,l])

        # compute second contribution
        tmp6 = np.einsum('djk,djl->kl', tmp5, self.B.tensor.conj())
        D += np.einsum('kl,m->lkm', tmp6, np.eye(D)[:,0])

        return D

        

if __name__ == "__main__":

    D, d = 8, 2
    A = UniformMps.random(D, d)

    from qdmt.model import TransverseFieldIsing
    import opt_einsum as oe

    model = TransverseFieldIsing(1.5, 0.1)
    U1 = model._compute_U_quarter_dt()
    U2 = model._compute_U_half_dt()

    # construct arbitrary vector
    v = np.random.rand(D, d, d, D) + 1j * np.random.rand(D, d, d, D)

    # scaling of computing _deriv_right_at()
    tensors = [v, A.tensor, A.tensor[:,:,0], U1[:,:,:,0], U2[:,:,:,0], U1, A.tensor.conj()[:,:,0]]
    path = oe.contract_path('abcd,aef,fg,egh,bhi,cijk,lk->djl', *tensors)
    print(path)

    tensors = [v, A.tensor, A.tensor[:,:,0], U1[:,:,:,0], U2[:,:,:,0], U1, A.tensor.conj(), np.eye(D)[:,0]]
    path = oe.contract_path('abcd,aef,fg,egh,bhi,cijk,djl,m->lkm', *tensors)
    print(path)

    # scaling of computing _left_at()
    tensors = (A.tensor[0,:,:], A.tensor, U1, U2[0,:,:,:], U1[0,:,:,:], A.tensor.conj()[0,:,:], A.tensor.conj())
    path = oe.contract_path('ab,bcd,acef,egh,gij,ik,kjl->dfhl', *tensors)
    print(path)

    # scaling of computing _rmatvec()
    tensors = (v, A.tensor, A.tensor, U1, U2, U1, A.tensor.conj(), A.tensor.conj())
    path = oe.contract_path('abcd,aef,fgh,egij,bikl,ckmn,dmo,onp->hjlp', *tensors)
    print(path)
