import pytest
from qdmt.uniform_mps import UniformMps
from qdmt.cost import EvolvedHilbertSchmidt
from qdmt.model import TransverseFieldIsing
from qdmt.fixed_point import RightFixedPoint

# @pytest.fixture(params=[2, 4, 6, 8, 10, 16])
# def L(request):
#     return request.param

# @pytest.mark.parametrize("L", [i for i in range(4, 100, 4)])
@pytest.fixture(params=[i for i in range(4, 200, 2)])
def f(request):
    A = UniformMps.new(6, 2)
    tfim = TransverseFieldIsing(0.2, 0.1)
    return EvolvedHilbertSchmidt(A, tfim, request.param, trotterization_order=2)

def test_cost(benchmark, f):
    B = f.A
    rB = RightFixedPoint.from_mps(B)
    benchmark(f.cost, B, rB)

# def test_deriv(benchmark, f):
#     B = f.A
#     rB = RightFixedPoint.from_mps(B)
#     benchmark(f.derivative, B, rB)