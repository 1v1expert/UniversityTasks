import numpy as np
import json

#import numpy as np
import contextlib


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

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
		self.top_game_price = None
		self.lower_game_price = None
		self.result_matrix = None
		self.winnings_ab = None
		self.get_price_game()
		
		self.static_matrix = np.array([[1, 1, 0, 0], [2,2,2,2]])
		self.winnings_a = np.array([np.ones(self.n)])
		self.winnings_b = np.array([np.ones(self.n)])
		
		self.building_payment_matrix()
		#ll = self.np_matrix.tolist()
		#print(ll.index([3, 0, 1]))
	
	def building_payment_matrix(self):
		for i, row in enumerate(self.static_matrix):
			#print(i, row)
			x1, y1 = row[0]-1, row[1]-1
			#print(self.np_matrix[:, x1])
			#print()
			self.winnings_a = np.vstack([self.winnings_a, self.np_matrix[:, x1]])
			self.winnings_b = np.vstack([self.winnings_b, self.np_matrix[y1, :]])
			if not i:
				self.winnings_a = np.delete(self.winnings_a, (0), axis=0) # remove first line
				self.winnings_b = np.delete(self.winnings_b, (0), axis=0)  # remove first line
			
		self.pprint_payment_matrix()
			#self.winnings_a.put(self.np_matrix.max(axis=0))
		#print(self.winnings_a, self.winnings_b)
		
	def pprint_payment_matrix(self):
		self.winnings_ab = np.column_stack((self.winnings_a, self.winnings_b))
		self.result_matrix = np.column_stack((self.winnings_ab, self.static_matrix))
		
		pre_line = 'Ax1\tAx2\tAx3\tBy1\tBy2\tBy3'
		print(pre_line)
		for i, row in enumerate(self.result_matrix):
			line = '\t'.join(['{}'.format(i) for i in row])
			print(line)

	
	def get_price_game(self):
		self.top_game_price = self.np_matrix.max(axis=0).min()
		self.lower_game_price = self.np_matrix.min(axis=1).max()
		return self.top_game_price, self.lower_game_price
		#print(price_game_b, price_game_a, self.top_game_price, self.lower_game_price)

		
		
if __name__ == '__main__':
	test_matrix = [[2, 1, 3], [3, 0, 1], [1, 2, 1]]
	AnalyticalGame(matrix=test_matrix, calculate=True)
	BraunRobinsGame(matrix=test_matrix, calculate=True)
	
#with printoptions(threshold=4, edgeitems=20, linewidth=80, suppress=True):
# print(x)