import GPy

class Prod(GPy.kern.Prod):
	"""
	This class overrides the original GPy.kern.Prod kernel to fix an edge-case bug.
	See https://github.com/SheffieldML/GPy/issues/568 for details on the bug.
	"""
	def __init__(self, kernels, name='mul'):
		unwrapped_kernels = []
		for kern in kernels:
			if isinstance(kern, GPy.kern.Prod):
				for part in kern.parts[::-1]:
					unwrapped_kernels.append(part.copy())
			else:
				unwrapped_kernels.append(kern.copy())
		super(GPy.kern.Prod, self).__init__(unwrapped_kernels, name)