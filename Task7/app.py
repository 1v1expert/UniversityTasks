
class AnalyticalGame(object):
	"""H(x,y) = a*x^2 + b*y^2 + c*x*y + d*x * e*y"""
	def __init__(self, **args):
		print(args)
	
	def _solve(self):
		pass


if '__main__' == __name__:
	matrix = {
		'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5
	}
	AnalyticalGame(**matrix)