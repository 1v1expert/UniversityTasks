import numpy as np

def print_matrix(matrix, matrix_name="Матрица:"):
	print(matrix_name)
	print("\n".join(", ".join("{0:.3f}".format(x) for x in row) for row in matrix))


class InformationalConfrontationGame(object):
	def __init__(self, dim=10, epsilon=10e-6, opinion_range=(0, 100), initial_opinions=None, trust_matrix=None):
		self.dim = dim
		self.epsilon = epsilon
		self.opinion_range = opinion_range
		self.initial_opinions = initial_opinions
		if not initial_opinions:
			self.gen_initial_opinions()
		
		self.trust_matrix = trust_matrix
		if not trust_matrix:
			self.gen_trust_matrix()
	
	def gen_initial_opinions(self):
		self.initial_opinions = np.random.randint(self.opinion_range[0], self.opinion_range[1], self.dim)
	
	def gen_trust_matrix(self):
		matrix = []
		for _ in np.arange(self.dim):
			row = np.random.sample(self.dim)
			matrix.append(row / row.sum())
		self.trust_matrix = np.array(matrix)
	
	def reach_accuracy(self, opinions):
		_iter = 0
		accuracy_reached = True
		while accuracy_reached:
			_iter += 1
			
			new_opinions = self.trust_matrix.dot(opinions).transpose()
			if all(x <= self.epsilon for x in np.abs(opinions - new_opinions)):
				accuracy_reached = False
				opinions = new_opinions
		return opinions, _iter
	
	def solve(self):
		result_opinions, iter_count = self.reach_accuracy(self.initial_opinions)
		print_matrix(self.trust_matrix, "Матрица доверия:")
		print("Изначальные мнения агентов:")
		print("X(0) =", self.initial_opinions)
		print("Потребовалось итераций:", iter_count)
		print("Результирующее мнение агентов (без влияния):")
		print("X(t->inf) =", ", ".join("{0:.3f}".format(x) for x in result_opinions))