import sys

from sympy import Rational

from kernel_function import KernelFunction
from analytical_solver import AnalyticalSolver
from base_solver import NoSolutionException
from numerical_solver import NumericalSolver


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


def main():
    kernel = KernelFunction(
        a=Rational(-10),
        b=Rational(15),
        c=Rational(60),
        d=Rational(-12),
        e=Rational(-48)
    )

    compute_analytical_solution(kernel)

    compute_numerical_solution(kernel)


if __name__ == '__main__':
    main()
