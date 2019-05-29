from Tasks.Task6L1.app import AnalyticalGame as AG, BraunRobinsGame as BG
from Tasks.Task7L2.app import OptimalStrategy as OG

if __name__ == '__main__':
	#test_matrix = [[2, 1, 3], [3, 0, 1], [1, 2, 1]]
	#matrix = [[17, 4, 9], [0, 16, 9], [12, 2, 19]]
	#BG(matrix=matrix, calculate=True, print=False)
	
	ponomorenko_matrix = {'a': -10, 'b': 15, 'c': float(60), 'd': float(-12), 'e': float(-48)}
	with OG(**ponomorenko_matrix) as game:
		print(game._get_h(x=0, y=0), game._get_h(x=1, y=0), game._get_h(x=0, y=1), game._get_h(x=1, y=1))
		matrix = game.build_step_matrix(54)
		
		print(matrix)
		BG(matrix=matrix, calculate=True, print=False)