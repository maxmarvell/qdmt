from abc import ABC, abstractmethod
import numpy as np
import copy
from qdmt.uniform_mps import UniformMps
from qdmt.transfer_matrices.base import RightFixedPoint

class AbstractCostFunction(ABC):

    @abstractmethod
    def cost(self, B: UniformMps, rB: RightFixedPoint) -> np.float64:
        pass

    @abstractmethod
    def derivative(self, B: UniformMps, rB: RightFixedPoint) -> np.ndarray:
        pass

    def copy(self) -> 'AbstractCostFunction':
        new_f = copy.copy(self)
        return new_f