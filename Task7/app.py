
class AnalyticalGame(object):
	"""H(x,y) = a*x^2 + b*y^2 + c*x*y + d*x * e*y"""
	def __init__(self, **args):
		[self.__setattr__(elem, args.get(elem)) for elem in args.keys()]
		
		Hxx = 2 * self.a
		Hyy = 2 * self.b
		
		if self.check_convex_concave(Hxx, Hyy):
			self._solve()
			
	@staticmethod
	def check_convex_concave(Hxx, Hyy):
		if (Hxx > 0 and Hyy < 0) or (Hxx < 0 and Hyy > 0):
			return True
		else:
			return False
	
	def _solve(self):
		pass


if '__main__' == __name__:
	matrix = {
		'a': -3, 'b': 3/2, 'c': 18/5, 'd': -18/50, 'e': -72/25
	}
	AnalyticalGame(**matrix)