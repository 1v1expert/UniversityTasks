from abc import ABC, abstractmethod
from typing import Tuple

from Tasks.Task7L2v2.kernel_function import KernelFunction


class NoSolutionException(Exception):
    pass


class BaseSolver(ABC):
    def __init__(self, kernel: KernelFunction):
        if kernel.dH_dx_dx >= 0:
            raise NoSolutionException(
                'Second derivative of Kernel function %s with respect to'
                ' x should be less than zero but it is %s',
                kernel.H,
                kernel.dH_dx_dx
            )

        if kernel.dH_dy_dy <= 0:
            raise NoSolutionException(
                'Second derivative of Kernel function %s with respect to '
                'y should be greater than zero but it is %s',
                kernel.H,
                kernel.dH_dy_dy
            )
        self.kernel = kernel

    @abstractmethod
    def solve(self) -> Tuple:
        pass
