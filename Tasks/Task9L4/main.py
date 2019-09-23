import numpy as np
from numpy.linalg import inv
from termcolor import colored


class SolutionAbstract(object):
    def __init__(self, bimatrix):
        self._matrix = bimatrix
        
    def check_optimality_nash(self, i, j):
        matrix_dim, _ = self._matrix[0].shape
        best_a_strategy = False
        best_b_strategy = False
        for iterI in range(matrix_dim):
            if self._matrix[0][iterI][j] > self._matrix[0][i][j]:
                best_b_strategy = True
            if self._matrix[1][i][iterI] > self._matrix[1][i][j]:
                best_a_strategy = True
    
        return not (best_a_strategy or best_b_strategy)
    
    def check_effectiveness_pareto(self, i, j):
        matrix_dim, _ = self._matrix[0].shape
        best_strategy = False
        for iIter in range(matrix_dim):
            for jIter in range(matrix_dim):
                if (
                        (self._matrix[0][iIter][jIter] > self._matrix[0][i][j] and
                         self._matrix[1][iIter][jIter] >= self._matrix[1][i][j]) or
                        (self._matrix[1][iIter][jIter] > self._matrix[1][i][j] and
                         self._matrix[0][iIter][jIter] >= self._matrix[0][i][j])
                ):
                    best_strategy = True
        return not best_strategy
    
    def print_analytical_result(self):
        print("Analytic method:")
        matrix_dim, _ = self._matrix[0].shape
        u = np.ones((1, matrix_dim), int)
        tmp_a = u.dot(inv(self._matrix[0]).dot(u.transpose()))
        tmp_b = u.dot(inv(self._matrix[0]).dot(u.transpose()))
        v_x = 1 / tmp_a
        v_y = 1 / tmp_b
        print("VX = {0:.2f}".format(v_x[0][0]))
        print("VY = {0:.2f}".format(v_y[0][0]))
        x = v_x * inv(self._matrix[0]).dot(u.transpose())
        y = v_y * u.dot(inv(self._matrix[0]))
        print("X = " + " ".join(["{0:.2f}".format(el) for el in x.transpose()[0]]))
        print("Y = " + " ".join(["{0:.2f}".format(el) for el in y[0]]))
        
    def print_result(self):
        matrix_dim, _ = self._matrix[0].shape
        for i in range(matrix_dim):
            line = []
            for j in range(matrix_dim):
                is_nash_opt = self.check_optimality_nash(i, j)
                is_pareto_eff = self.check_effectiveness_pareto(i, j)
                point_str = "({0: <3}/{1: <3})".format(self._matrix[0][i][j], self._matrix[1][i][j])
                if is_pareto_eff and is_nash_opt:
                    line.append(colored(point_str, 'red'))
                elif is_pareto_eff:
                    line.append(colored(point_str, 'green'))
                elif is_nash_opt:
                    line.append(colored(point_str, 'blue'))
                else:
                    line.append(point_str)
            print(" ".join(line))
        print()


if __name__ == "__main__":
    crosswayBiMatrix = np.array([[[1, 0.5], [2, 0]], [[1, 2], [0.5, 0]], ], float)
    familyDisputeBiMatrix = np.array([[[4, 0], [0, 1]], [[1, 0], [0, 4]], ], float)
    prisonerDilemmaBiMatrix = np.array([[[-5, 0], [-10, -1]], [[-5, -10], [0, -1]], ], float)
    MAX_VALUE = 50
    MIN_VALUE = 0
    DIM = 10
    randomBiMatrix = np.array([np.random.randint(MIN_VALUE, MAX_VALUE, (DIM, DIM)),
                               np.random.randint(MIN_VALUE, MAX_VALUE, (DIM, DIM)), ], int)
    variant3BiMatrix = np.array([[[3, 5], [9, 2]], [[1, 0], [6, 3]], ], float)
    print(colored('Парето эффективность и равновесие по Нэшу.', 'red'))
    print(colored('Парето эффективность.', 'green'))
    print(colored('Равновесие по Нэшу.', 'blue'))
    print()
    print("Перекресток со смещением:")
    game = SolutionAbstract(crosswayBiMatrix)
    game.print_result()
    print("Семейный спор:")
    game = SolutionAbstract(familyDisputeBiMatrix)
    game.print_result()
    print("Дилемма заключенного:")
    game = SolutionAbstract(prisonerDilemmaBiMatrix)
    game.print_result()
    print("Случайная матрица 10х10:")
    game = SolutionAbstract(randomBiMatrix)
    game.print_result()
    print("Матрица варианта №3:")
    game = SolutionAbstract(variant3BiMatrix)
    game.print_result()
    game.print_analytical_result()

