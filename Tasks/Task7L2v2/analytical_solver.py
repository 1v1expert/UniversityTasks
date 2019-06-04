from typing import Tuple

from sympy import solveset, Eq, Rational

from kernel_function import KernelFunction
from base_solver import BaseSolver


class AnalyticalSolver(BaseSolver):
    def __init__(self, kernel: KernelFunction):
        super().__init__(kernel)

    def solve(self) -> Tuple[Rational, Rational, Rational]:
        x_expression = list(
            solveset(Eq(self.kernel.dH_dx, 0), self.kernel.x)
        )[0]
        y_expression = list(
            solveset(Eq(self.kernel.dH_dy, 0), self.kernel.y)
        )[0]

        y_expression_without_x = y_expression.subs(
            self.kernel.x, x_expression)
        y_solution = list(
            solveset(
                Eq(y_expression_without_x, self.kernel.y),
                self.kernel.y
            )
        )[0]

        x_solution = x_expression.subs(self.kernel.y, y_solution)

        return x_solution, y_solution, self.kernel.get_value_at_point(
            x_solution,
            y_solution
        )
