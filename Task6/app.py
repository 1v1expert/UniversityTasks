import numpy as np
import json

class Game(object):
	
	def __init__(self, **args):
		#print(args)
		self.matrix = args.get('matrix', None)
		self.calculate = args.get('calculate', None)
		
		self.n = len(self.matrix) #lines
		self.ones = np.ones(self.n)
		self.np_matrix = np.array(self.matrix, dtype=np.float)
		self.m = self.np_matrix.size / self.n  # columns
		self.inv_matrix = np.linalg.inv(self.np_matrix)
		self.optimal_strategy_x, self.optimal_strategy_y = None, None
		
		if self.calculate:
			print(self._get_price_game())
		
		#with np.printoptions(precision=3, suppress=True):
		#	print(vars(self))
		#print(json.dumps(vars(self), indent=4, sort_keys=True))
		#print(s)
	
	def _get_price_game(self):
		self.denominator = np.dot(np.dot(self.ones, self.inv_matrix),  self.ones.transpose())
		self.optimal_strategy_x = np.dot(self.inv_matrix, self.ones.transpose()) / self.denominator
		self.optimal_strategy_y = np.dot(self.ones, self.inv_matrix) / self.denominator
		return self.optimal_strategy_x, self.optimal_strategy_y, 1 / self.denominator
	
if __name__ == '__main__':
	matrix = [[2, 1, 3], [3, 0, 1], [1, 2, 1]]
	#print(len(matrix))
	Game(matrix=matrix, calculate=True)
	
	