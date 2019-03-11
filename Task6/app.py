import numpy as np
import json

class AnalyticalGame(object):
	
	def __init__(self, **args):
		self.matrix = args.get('matrix', None)
		self.calculate = args.get('calculate', None)
		self.print = args.get('print', True)
		
		self.n = len(self.matrix) #lines
		self.ones = np.ones(self.n)
		self.np_matrix = np.array(self.matrix, dtype=np.float)
		self.m = self.np_matrix.size / self.n  # columns
		self.inv_matrix = np.linalg.inv(self.np_matrix)
		self.optimal_strategy_x, self.optimal_strategy_y = None, None
		
		if self.calculate:
			self.optimal_strategy_x, self.optimal_strategy_y, self.price = self._get_decision_game()

	def _get_decision_game(self):
		self.denominator = np.dot(np.dot(self.ones, self.inv_matrix),  self.ones.transpose())
		x = np.dot(self.inv_matrix, self.ones.transpose()) / self.denominator
		y = np.dot(self.ones, self.inv_matrix) / self.denominator
		return x, y, 1 / self.denominator
	
	def print_result(self):
		if self.print:
			print('Optimal strategy X: {},'\
				'\nOptimal strategy Y: {},'
				'\nPrice game = {}'.format(self.optimal_strategy_x,
			                               self.optimal_strategy_y,
			                               1/self.denominator)
				)
		
	
	
class BraunRobinsGame(AnalyticalGame):
	
	def __init__(self, **args):
		super().__init__(**args)
		price_game_b = self.np_matrix.max(axis=0)
		price_game_a = self.np_matrix.min(axis=1)
		ll = self.np_matrix.tolist()
		print(ll.index([3, 0, 1]))
		self.top_game_price = price_game_b.min()
		self.lower_game_price = price_game_a.max()
		print(price_game_b, price_game_a, self.top_game_price, self.lower_game_price)
		
		first = [1, 1, 0, 0]
		spisok = []
		
		
if __name__ == '__main__':
	matrix = [[2, 1, 3], [3, 0, 1], [1, 2, 1]]
	#AnalyticalGame(matrix=matrix, calculate=True)
	BraunRobinsGame(matrix=matrix, calculate=True)
	
	