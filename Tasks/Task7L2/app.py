import numpy as np

import sys


class AnalyticalGame(object):
    ''' Аналитический(матричный) метод решения матричной игры с нулевой суммой '''
    
    def __init__(self, **args):
        self.matrix = args.get('matrix', None)
        self.calculate = args.get('calculate', None)
        self.print = args.get('print', True)
        
        self.n = len(self.matrix)  # lines
        self.ones = np.ones(self.n)
        self.np_matrix = np.array(self.matrix, dtype=np.float)
        self.m = self.np_matrix.size / self.n  # columns
        self.inv_matrix = np.linalg.inv(self.np_matrix)
        self.optimal_strategy_x, self.optimal_strategy_y = None, None
        
        if self.calculate:
            self.optimal_strategy_x, self.optimal_strategy_y, self.price = self._get_decision_game()
        
        self.print_result()
    
    def _get_decision_game(self):
        self.denominator = np.dot(np.dot(self.ones, self.inv_matrix), self.ones.transpose())
        x = np.dot(self.inv_matrix, self.ones.transpose()) / self.denominator
        y = np.dot(self.ones, self.inv_matrix) / self.denominator
        return np.around(x, decimals=2), np.around(y, decimals=2), 1 / self.denominator
    
    def print_result(self):
        if self.print:
            print('Optimal strategy X: {},' \
                  '\nOptimal strategy Y: {},'
                  '\nPrice game = {}'.format(self.optimal_strategy_x,
                                             self.optimal_strategy_y,
                                             1 / self.denominator)
                  )


class SolveIteration(object):
    
    def __init__(self):
        pass


class OptimalStrategy(object):
    
    def __init__(self, **kwargs):
        [self.__setattr__(elem, float(kwargs.get(elem))) for elem in kwargs.keys()]
        # [self.__setattr__(elem, args[i]) for i, elem in enumerate(('a', 'b', 'c', 'd', 'e'))]
        Hxx = 2 * self.a
        Hyy = 2 * self.b
        self.x, self.y, self.h = None, None, None
        
        if self.check_convex_concave(Hxx, Hyy):
            self._solve_lin()
        # self._solve_lin_precision()
    
    def __enter__(self, **kwargs):
        self.__init__(**kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('X: {}, Y: {}, H: {}'.format(self.x, self.y, self.h))
    
    def _solve_lin_precision(self):
        """
		y = (2*a*e - c*d)/(c^2 - 4*a*b)
		x = -(c*y + d)/(2*a)
		"""
        self.y = (2 * self.a * self.e - self.c * self.d) / (self.c * self.c - 4 * self.a * self.b)
        self.x = - (self.c * self.y + self.d) / (2 * self.a)
        self.h = self._get_h()
    
    # print(y, x, self.H(x, y))
    
    def _solve_lin(self):
        """Solve the system of equations dH(x)/d(x) = 2 * a * x + c * y + d and dH(y)/d(y) = 2 * b * y + c * x + e """
        Hx = [2 * self.a, self.c]
        Hy = [2 * self.b, self.c]
        system_equations = np.array([Hx, Hy], dtype=np.float)
        zeros = np.array([-self.d, -self.e], dtype=np.float)
        self.x, self.y = np.linalg.solve(system_equations, zeros)
        self.h = self._get_h()
    
    def _get_h(self, x=None, y=None):
        """H(x,y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y"""
        if not (x != None or y != None):
            x, y = self.x, self.y
        
        return self.a * pow(x, 2) + self.b * pow(y, 2) + self.c * x * y + self.d * x + self.e * y
    
    @staticmethod
    def check_convex_concave(Hxx, Hyy):
        if (Hxx > 0 and Hyy < 0) or (Hxx < 0 and Hyy > 0):
            return True
        else:
            return False
    
    @staticmethod
    def build_metric(n):
        step = 1 / n
        first_step = 0
        return [first_step + i * step for i in range(n + 1)]
    
    def build_step_matrix(self, n):
        gradation = self.build_metric(n)
        matrix = []
        for i in range(n + 1):
            line = []
            for j in range(n + 1):
                value = self._get_h(gradation[i], gradation[j])
                line.append(value)
            matrix.append(line)
        # print(matrix)
        return matrix
    
    def _solve(self):
        pass
    
    def custom_solve_equation(self):
        pass


if '__main__' == __name__:
    matrix = {
        'a': -3, 'b': 3 / 2, 'c': float(18 / 5), 'd': float(-18 / 50), 'e': float(-72 / 25)
    }
    ponomorenko_matrix = {
        'a': -10, 'b': 15, 'c': float(60), 'd': float(-12), 'e': float(-48)
    }
    
    with AnalyticalGame(**ponomorenko_matrix) as game:
        print(game._get_h(0.5, 0))
        matrix = game.build_step_matrix(2)
        print(matrix)
    print(sys.path)
