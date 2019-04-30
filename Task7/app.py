import numpy as np


class AnalyticalGame(object):
	
	def __init__(self, **args):
		[self.__setattr__(elem, float(args.get(elem))) for elem in args.keys()]
		#[self.__setattr__(elem, args[i]) for i, elem in enumerate(('a', 'b', 'c', 'd', 'e'))]
		Hxx = 2 * self.a
		Hyy = 2 * self.b
		
		if self.check_convex_concave(Hxx, Hyy):
			self._solve_lin()
			#self._solve_lin_precision()
	
	def _solve_lin_precision(self):
		"""
		y = (2*a*e - c*d)/(c^2 - 4*a*b)
		x = -(c*y + d)/(2*a)
		"""
		y = (2 * self.a * self.e - self.c * self.d)/(self.c * self.c - 4 * self.a * self.b)
		x = - (self.c * y + self.d)/(2*self.a)
		print(y, x, self.H(x, y))
		
	def _solve_lin(self):
		"""Solve the system of equations dH(x)/d(x) = 2 * a * x + c * y + d and dH(y)/d(y) = 2 * b * y + c * x + e """
		Hx = [2 * self.a, self.c]
		Hy = [2 * self.b, self.c]
		system_equations = np.array([Hx, Hy], dtype=np.float)
		zeros = np.array([-self.d, -self.e], dtype=np.float)
		x, y = np.linalg.solve(system_equations, zeros)
		print('X: {}, Y: {}, H: {}'.format(x, y, self.H(x, y)))
	
	def H(self, x, y):
		"""H(x,y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y"""
		return self.a * pow(x, 2) + self.b * pow(y, 2) + self.c * x * y + self.d * x + self.e * y
		
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
	ponomorenko_matrix = {
		'a': -10, 'b': 15, 'c': float(60), 'd': float(-12), 'e': float(-48)
	}
	#print('y= ', )
	#from decimal import *
	# matrix = [-3, 3 / 2, 18 / 5, -18 / 50, -72 / 25]
	instance = AnalyticalGame(**ponomorenko_matrix)
	print(instance.H(0, 1))