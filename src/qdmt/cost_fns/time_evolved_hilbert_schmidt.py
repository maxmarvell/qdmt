from qdmt.cost_fns.base import AbstractCostFunction
from qdmt.uniform_mps import UniformMps
from qdmt.model import AbstractModel
from qdmt.transfer_matrices import SecondOrder, TransferMatrix, RightFixedPoint
import numpy as np
from qdmt.utils.mps import trotter_step
from ncon import ncon

class TimeEvolvedHilbertSchmidt(AbstractCostFunction):

    def __init__(self, A: UniformMps, model: AbstractModel, L: int, *, trotterization_order: int = 2):

        self.L, self.A = L, A

        E_A = TransferMatrix(A, A)
        self.rA = E_A.right_fixed_point()

        self.order = trotterization_order

        if self.order != 2:
            raise NotImplementedError("TimeEvolvedHilbertSchmidt currently only implemented for 2nd order trotterization only.")
        
        if L % 2 == 1:
            raise NotImplementedError("Second-order Trotterization is not implemented for odd L.")
        
        self.U1, self.U2 = model.trotter_second_order()

        self.purity_A = self._compute_second_trotterized_purity()

        self._precompute_entangling_space_tensors()
        

    def _compute_second_trotterized_purity(self) -> np.complex128:

        A, U1, U2, L, rA = self.A, self.U1, self.U2, self.L, self.rA

        if L <= 4:

            N = L + 4

            mps = A.to_mps_chain(N)

            # trotterized evolve
            mps = trotter_step(mps, U1)
            mps = trotter_step(mps, U2, start=1, stop=2)
            mps = trotter_step(mps, U2, start=N-3, stop=N-2)

            # compute reduced density matrix over patch size L
            tensors = [mps, mps.conj(), rA.tensor]
            indices = [
                [1, 2, 3] + [-(i+1) for i in range(L)] + [5, 6, 7],
                [1, 2, 3] + [-(i+1+L) for i in range(L)] + [5, 6, 8],
                [7, 8]
            ]
            rho_A = ncon(tensors, indices)

            # compute trace product
            tensors = [rho_A, rho_A]
            indices = [
                [i for i in range(1, 2*L+1)],
                [i for i in range(L+1, 2*L+1)] + [i for i in range(1, L+1)]
            ]
            return ncon(tensors, indices)
        
        else: 

            # create a base mps to calculate the left and right mixing
            mps = A.to_mps_chain(4)
            mps = trotter_step(mps, U1)
            mps = trotter_step(mps, U2, start=1)

            tensors = [mps, mps.conj()]
            indices = [[1, 2, 3, -4, -5, -6], [1, 2, 3, -3, -2, -1]]
            l = ncon(tensors, indices)

            tensors = [mps, mps.conj(), rA.tensor]
            indices = [[-1, -2, -3, 1, 2, 3], [-6, -5, -4, 1, 2, 4], [3, 4]]
            r = ncon(tensors, indices)

            E = TransferMatrix(A, A)
            purity = 0 + 0j

            # perform manual trace
            for i in range(self.A.D):
                for j in range(self.A.D):

                    v = E._left_at(i,j)

                    for _ in range(self.L-5):
                        v = E._rmatvec(v)

                    tensors = [v, r, r]
                    indices = [
                        [1, 2],
                        [1, 3, 4, 5, 6, -2],
                        [-1, 6, 5, 4, 3, 2]
                    ]
                    v = ncon(tensors, indices)

                    for _ in range(self.L-4):
                        v = E._matvec(v)
                    
                    tensors = [v, l[:,:,:,:,:,i], l[j,:,:,:,:,:]]
                    indices = [
                        [1, 2],
                        [2, 3, 4, 5, 6],
                        [6, 5, 4, 3, 1]
                    ]
                    purity += ncon(tensors, indices)

            return purity
        
    def _compute_variational_purity(self, B: UniformMps, rB: RightFixedPoint) -> np.complex128:
        E = TransferMatrix(B, B)
        purity = 0 + 0j
        for i in range(B.D):
            for j in range(B.D):
                v = E._left_at(i, j)
                for _ in range(self.L - 1):
                    v = E._rmatvec(v)
                v = np.einsum('ab,ac,bd->dc', v, rB.tensor, rB.tensor.T)
                for _ in range(self.L):
                    v = E._matvec(v)
                purity += v[j,i]
        return purity

    def _compute_reduced_overlap(self, B: UniformMps, rB: RightFixedPoint) -> np.complex128:
        
        overlap = 0 + 0j
        AB = SecondOrder(self.A, B, self.U1, self.U2)
        BA = SecondOrder(B, self.A, U1.conj().transpose(2, 3, 0, 1), U2.conj().transpose(2, 3, 0, 1))

        Da, d, Db = self.A.D, self.A.d, B.D
        for i in range(Da):
            for j in range(d):
                for k in range(d):
                    for l in range(Db):
                        v = AB._left_at(i,j,k,l)
                        for i in range(self.L//2):
                            v = AB._rmatvec(v)


    # def _contract_right_entangling_space(self, v: np.ndarray, rB: RightFixedPoint):

    #     fabc = np.einsum('df,abcd->fabc', rB.tensor.T, v)
    #     hga = np.einsum('hge,ae->hga', self.A.tensor.conj(), self.rA.tensor)
    #     gani = np.einsum('hga,nih->gani', hga, self.A.tensor.conj())


                        
    def _precompute_entangling_space_tensors(self) -> np.complex128:
        pass


    def cost():
        pass

    def derivative(self, B, rB):
        pass

if __name__ == "__main__":

    D, d = 12, 2
    A = UniformMps.random(D, d)
    rA = TransferMatrix(A, A).right_fixed_point()

    from qdmt.model import TransverseFieldIsing
    import opt_einsum as oe

    model = TransverseFieldIsing(1.5, 0.1)
    U1 = model._compute_U_quarter_dt()
    U2 = model._compute_U_half_dt()

    # construct arbitrary vector
    v = np.random.rand(D, d, d, D) + 1j * np.random.rand(D, d, d, D)

    f = TimeEvolvedHilbertSchmidt(A, model, 12)
    f._compute_second_trotterized_purity()

    # check contraction order for right entangling space contraction
    tensors = (v.shape, rA.tensor, rA.tensor.T, A.tensor.conj(), A.tensor.conj(), U1, U2, U1, A.tensor.conj(), A.tensor.conj())
    constants = [1,2,3,4,5,6,7,8,9]
    path = oe.contract_expression('abcd,ae,df,hge,nih,jbig,kclj,mlrq,oqn,pro->fkmp', *tensors, constants=constants)
    print(path)

    # Get the path info
    tensors = (v, rA.tensor, rA.tensor.T, A.tensor.conj(),
    A.tensor.conj(),
                U1, U2, U1, A.tensor.conj(), A.tensor.conj())
    constants = [1,2,3,4,5,6,7,8,9]

    path, path_info = oe.contract_path(
        'abcd,ae,df,hge,nih,jbig,kclj,mlrq,oqn,pro->fkmp',
        *tensors,
        optimize='optimal'
    )

    # Print detailed complexity info
    print(path_info)

