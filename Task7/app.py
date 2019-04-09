import numpy as np


class AnalyticalGame(object):
	"""H(x,y) = a*x^2 + b*y^2 + c*x*y + d*x * e*y"""
	def __init__(self, **args):
		[self.__setattr__(elem, float(args.get(elem))) for elem in args.keys()]
		#[self.__setattr__(elem, args[i]) for i, elem in enumerate(('a', 'b', 'c', 'd', 'e'))]
		Hxx = 2 * self.a
		Hyy = 2 * self.b
		
		if self.check_convex_concave(Hxx, Hyy):
			self._solve_lin()
			
	def _solve_lin(self):
		"""Solve the system of equations dH(x)/d(x) = 2 * a * x + c * y + d and dH(y)/d(y) = 2 * b * y + c * x + e """
		Hx = [2 * self.a, self.c]
		Hy = [2 * self.b, self.c]
		system_equations = np.array([Hx, Hy], dtype=np.float)
		print(vars(self))
		zeros = np.array([-self.d, -self.e], dtype=np.float)
		x, y = np.linalg.solve(system_equations, zeros)
		#H =
		#x , y = solve
		print('X: {}, Y: {}, H: {}'.format(x, y, self.H(x, y)))
	
	def H(self, x, y):
		return self.a * pow(x, 2) + self.b * pow(y, 2) + self.c * x * y + self.d * x * self.e * y
		
	@staticmethod
	def check_convex_concave(Hxx, Hyy):
		if (Hxx > 0 and Hyy < 0) or (Hxx < 0 and Hyy > 0):
			return True
		else:
			return False
	
	def _solve(self):
		pass
	
	def custom_solve_equation(self):
		pass

if '__main__' == __name__:
	matrix = {
		'a': -3, 'b': 3/2, 'c': float(18/5), 'd': float(-18/50), 'e': float(-72/25)
	}
	#from decimal import *
	# matrix = [-3, 3 / 2, 18 / 5, -18 / 50, -72 / 25]
	AnalyticalGame(**matrix)