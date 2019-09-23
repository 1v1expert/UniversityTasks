import sys
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Any, List, Optional
from collections import defaultdict

from sympy import symbols, Rational, diff, Eq, solveset, Matrix
from sympy.core.add import Add


class KernelFunction:
    def __init__(
        self,
        a: Rational = Rational(0),
        b: Rational = Rational(0),
        c: Rational = Rational(0),
        d: Rational = Rational(0),
        e: Rational = Rational(0),
    ):
        self.x, self.y = symbols('x y')
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    @property
    def H(self) -> Add:
        return (self.a * self.x ** 2) + (self.b * self.y ** 2) + \
               (self.c * self.x * self.y) + \
               (self.d * self.x) + (self.e * self.y)

    @property
    def dH_dx(self) -> Add:
        return diff(self.H, self.x)

    @property
    def dH_dx_dx(self) -> Add:
        return diff(self.dH_dx, self.x)

    @property
    def dH_dy(self) -> Add:
        return diff(self.H, self.y)

    @property
    def dH_dy_dy(self) -> Add:
        return diff(self.dH_dy, self.y)

    def get_value_at_point(self, x: Rational, y: Rational):
        return self.H.subs(self.x, x).subs(self.y, y)


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
            solveset(Eq(y_expression_without_x, self.kernel.y), self.kernel.y))[0]

        x_solution = x_expression.subs(self.kernel.y, y_solution)

        return x_solution, y_solution, self.kernel.get_value_at_point(
            x_solution,
            y_solution
        )


class BraunRobinsonAlgorithmStep:
    def __init__(
            self,
            a_choice: int,
            b_choice: int,
            payoff_matrix: Matrix,
            previous_step: Optional):
        self.step_number = 1
        if previous_step:
            self.step_number = previous_step.step_number + 1

        self.a_choice = a_choice
        self.b_choice = b_choice

        self.a_gain = payoff_matrix.col(self.b_choice).T.tolist()[0]
        self.b_gain = payoff_matrix.row(self.a_choice).tolist()[0]
        self.min_upper_game_cost = self.upper_game_cost
        self.max_lower_game_cost = self.lower_game_cost

        if previous_step:
            self.a_gain = [sum(x) for x in
                           zip(self.a_gain, previous_step.a_gain)]
            self.b_gain = [sum(x) for x in
                           zip(self.b_gain, previous_step.b_gain)]
            self.min_upper_game_cost = min(
                (self.upper_game_cost, previous_step.min_upper_game_cost))
            self.max_lower_game_cost = max(
                (self.lower_game_cost, previous_step.max_lower_game_cost))

    @property
    def upper_game_cost(self) -> Rational:
        return max(self.a_gain) / Rational(self.step_number)

    @property
    def lower_game_cost(self) -> Rational:
        return min(self.b_gain) / Rational(self.step_number)

    @property
    def epsilon(self) -> Rational:
        return self.min_upper_game_cost - self.max_lower_game_cost

    def __str__(self):
        return '{step_number:^5}|{a_choice:^3}|{b_choice:^3}|{a_gain:^25}'\
               '|{b_gain:^25}|{lower_game_cost:^12}|{upper_game_cost:^12}'\
               '|{epsilon:^8}'.format(
                step_number=self.step_number,
                a_choice=self.a_choice,
                b_choice=self.b_choice,
                a_gain=str(self.a_gain),
                b_gain=str(self.b_gain),
                lower_game_cost=str(self.lower_game_cost),
                upper_game_cost=str(self.upper_game_cost),
                epsilon=str(float(self.epsilon))[:8],
            )


class BraunRobinsonTable:
    def __init__(self, payoff_matrix: Matrix, first_a: int, first_b: int):
        self.payoff_matrix = payoff_matrix
        self.steps = []
        
        self.make_step(first_a, first_b)
    
    def solve(self, threshold: float = 0.01, max_steps: int = 2 ** 64):
        while threshold < self.steps[-1].epsilon \
                and max_steps > len(self.steps):
            self.make_step()
    
    def make_step(self, a_strategy: int = None, b_strategy: int = None):
        if all([a_strategy is None, b_strategy is None]):
            a_strategy = self._get_next_a_strategy()
            b_strategy = self._get_next_b_strategy()
        elif any([a_strategy is None, b_strategy is None]):
            raise Exception(
                'Both strategy should be defined or should be None.'
                ' Got strategies: %s',
                [a_strategy, b_strategy]
            )
        
        step = BraunRobinsonAlgorithmStep(
            a_choice=a_strategy,
            b_choice=b_strategy,
            payoff_matrix=self.payoff_matrix,
            previous_step=self.steps[-1] if self.steps else None
        )
        self.steps.append(step)
    
    def get_a_mixed_strategy(self):
        counter = defaultdict(int)
        for step in self.steps:
            counter[step.a_choice] += 1
        
        return [Rational(counter[choice], len(self.steps))
                for choice in range(self.payoff_matrix.cols)]
    
    def get_b_mixed_strategy(self):
        counter = defaultdict(int)
        for step in self.steps:
            counter[step.b_choice] += 1
        
        return [Rational(counter[choice], len(self.steps))
                for choice in range(self.payoff_matrix.cols)]
    
    @staticmethod
    def annotate_table():
        return '{step_number:^5} {a_choice:^3} {b_choice:^3} {a_gain:^25}' \
               ' {b_gain:^25} {lower_game_cost:^12} {upper_game_cost:^12}' \
               ' {epsilon:^8}'.format(
            step_number='step',
            a_choice='A',
            b_choice='B',
            a_gain='A gain',
            b_gain='B gain',
            lower_game_cost='v lower',
            upper_game_cost='v upper',
            epsilon='E',
        )
    
    def _get_next_a_strategy(self) -> int:
        max_from_previous_a_gain = max(self.steps[-1].a_gain)
        
        if len(self.steps) > 1 and self.steps[-2].a_gain[
            self.steps[-1].a_choice] == max_from_previous_a_gain:
            return self.steps[-1].a_choice
        
        return self.steps[-1].a_gain.index(max_from_previous_a_gain)
    
    def _get_next_b_strategy(self) -> int:
        min_from_previous_b_gain = min(self.steps[-1].b_gain)
        
        if len(self.steps) > 1 and self.steps[-2].b_gain[
            self.steps[-1].b_choice] == min_from_previous_b_gain:
            return self.steps[-1].b_choice
        
        return self.steps[-1].b_gain.index(min_from_previous_b_gain)


class NumericalSolver(BaseSolver):
    def __init__(self, kernel: KernelFunction, steps: int):
        super().__init__(kernel)
        self.steps = steps
        self.grid_matrix = self.create_grid_matrix(self.steps)
    
    def solve(self) -> Tuple[float, float, Any]:
        shift = min([x for x in self.grid_matrix])
        if shift < 0:
            for i in range(self.grid_matrix.rows):
                for j in range(self.grid_matrix.cols):
                    self.grid_matrix[i, j] += -1 * shift
        
        maximin = self.get_maximin(self.grid_matrix)
        minimax = self.get_minimax(self.grid_matrix)
        
        if maximin == minimax:
            print('Есть седловая точка:')
            a_clear_strategy = maximin[1][0]
            b_clear_strategy = minimax[1][1]
        
        else:
            print('Седловой точки нет, решение методом Брауна-Робинсона:')
            braun_robinson_table = BraunRobinsonTable(self.grid_matrix, maximin[1][0], minimax[1][1])
            braun_robinson_table.solve(0.01)
            
            a_mixed_strategy = braun_robinson_table.get_a_mixed_strategy()
            b_mixed_strategy = braun_robinson_table.get_b_mixed_strategy()
            
            a_clear_strategy = \
                max([(a_mixed_strategy[i], i) for i in range(len(a_mixed_strategy))], key=lambda x: x[0])[1]
            
            b_clear_strategy = \
                max([(b_mixed_strategy[j], j) for j in range(len(b_mixed_strategy))], key=lambda x: x[0])[1]
        
        H = self.grid_matrix[a_clear_strategy, b_clear_strategy] + shift
        x = a_clear_strategy / self.steps
        y = b_clear_strategy / self.steps
        
        return x, y, H
    
    def create_grid_matrix(self, steps: int) -> Matrix:
        res = []
        for i in range(steps + 1):
            res.append(
                [self.kernel.get_value_at_point(Rational(i, steps), Rational(j, steps)) for j in range(steps + 1)])
        return Matrix(res)
    
    def matrix_to_str(self) -> str:
        res = '['
        for i in range(self.grid_matrix.rows):
            res += ' ['
            res += ', '.join(["{:8.3f}".format(float(i)) for i in self.grid_matrix.row(i)])
            res += ']\n'
        res = res[:-1] + ']'
        return res[:1] + res[2:]
    
    def get_maximin(self, payoff_matrix: Matrix) -> Tuple:
        mins_by_rows = self._find_extremums_by_axis(payoff_matrix, 'rows', fn=min, )
        maximum = max(mins_by_rows, key=lambda x: x[0])
        
        return maximum[0], (mins_by_rows.index(maximum), maximum[1])
    
    def get_minimax(self, payoff_matrix: Matrix) -> Tuple:
        maxs_by_cols = self._find_extremums_by_axis(payoff_matrix, 'columns', fn=max, )
        minimum = min(maxs_by_cols, key=lambda x: x[0])
        
        return minimum[0], (minimum[1], maxs_by_cols.index(minimum))
    
    def _find_extremums_by_axis(self, payoff_matrix: Matrix, axis: str, fn: Callable) -> List[Tuple]:
        if axis == 'columns':
            lines = [payoff_matrix.col(j).T.tolist()[0] for j in range(payoff_matrix.cols)]
        elif axis == 'rows':
            lines = [payoff_matrix.row(i).tolist()[0] for i in range(payoff_matrix.rows)]
        else:
            raise Exception("Axis should be 'rows' or 'columns'")
        
        return [(fn(line), line.index(fn(line))) for line in lines]


def compute_analytical_solution(kernel_function: KernelFunction):
    try:
        analytical_solver = AnalyticalSolver(kernel_function)
    except NoSolutionException as exception:
        print(exception)
        sys.exit(0)

    x, y, H = analytical_solver.solve()

    print('АНАЛИТИЧЕСКОЕ РЕШЕНИЕ')
    print('x={:2.3f} y={:2.3f} H={:2.3f}\n\n'.format(
        float(x), float(y), float(H))
    )
    
    
def compute_numerical_solution(kernel_function: KernelFunction):
    print('ЧИСЛЕННОЕ РЕШЕНИЕ')
    try:
        results = []
        steps = 1
        steps_shift = 10
        while len(results) < steps_shift or abs(
                    max(results[-1*steps_shift:]) -
                    min(results[-1*steps_shift:])
                ) > 0.01:
            print(f'N={steps}')
            numerical_solver = NumericalSolver(kernel_function, steps)

            if steps <= 10:
                print(numerical_solver.matrix_to_str())

            x, y, H = numerical_solver.solve()
            results.append(H)
            print('x={:2.3f} y={:2.3f} H={:2.3f}\n\n'.format(
                float(x), float(y), float(H))
            )

            steps += 1

    except NoSolutionException as exception:
        print(exception)
        sys.exit(0)


def process_solutions(kernel):
    if not kernel:
        return
    
    compute_analytical_solution(kernel)
    compute_numerical_solution(kernel)


def main(*args):
    kernel = KernelFunction(
        a=Rational(-10),
        b=Rational(15),
        c=Rational(60),
        d=Rational(-12),
        e=Rational(-48)
    )
    
    process_solutions(kernel)


if __name__ == '__main__':
    main()
