from Task6.app import AnalyticalGame, BraunRobinsGame

if __name__ == '__main__':
	test_matrix = [[2, 1, 3], [3, 0, 1], [1, 2, 1]]
	matrix = [[17, 4, 9], [0, 16, 9], [12, 2, 19]]
	BraunRobinsGame(matrix=matrix, calculate=True, print=False)