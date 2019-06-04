from typing import Callable, Tuple, Any, List

from sympy import Rational, Matrix

from lab7.braun_robinson.table import BraunRobinsonTable
from lab7.kernel_function import KernelFunction
from lab7.solvers.base_solver import BaseSolver


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
