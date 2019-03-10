import numpy as np

class Game(object):
	
	def __init__(self, **args):
		#print(args)
		self.ones = np.ones(1)
		self.m = args.get('m', None) #columns
		self.n = args.get('n', None) #lines
		self.matrix = np.array(args.get('matrix'))
		print(self.matrix)
	
	def price_game(self):
	
	
if __name__ == '__main__':
	Game(matrix=[[1.4, 5]])