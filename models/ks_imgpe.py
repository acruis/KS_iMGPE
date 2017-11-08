from imgpe import iMGPE
import numpy as np
import math
import GPy
import json
from GPy.kern import Add
from prod import Prod

class KSiMGPE(iMGPE):

	# --------------------
	# Additional kernels
	# --------------------

	def create_periodic_kernel(self):
		kern = GPy.kern.StdPeriodic(input_dim=1,
				variance=self.priors['periodic_variance'].rvs(1),
				lengthscale=self.priors['periodic_lengthscale'].rvs(1),
				period=self.priors['periodic_period'].rvs(1))
		kern.variance.set_prior(self.priors['periodic_variance'], warning=False)
		kern.lengthscale.set_prior(self.priors['periodic_lengthscale'], warning=False)
		kern.period.set_prior(self.priors['periodic_period'], warning=False)
		return kern

	def create_linear_kernel(self):
		kern = GPy.kern.Linear(input_dim=1, variances=self.priors['linear_variance'].rvs(1))
		kern.variances.set_prior(self.priors['linear_variance'], warning=False)
		return kern

	def create_constant_kernel(self):
		kern = GPy.kern.Bias(input_dim=1., variance=self.priors['constant_variance'].rvs(1))
		kern.variance.set_prior(self.priors['constant_variance'], warning=False)
		return kern

	# --------------------------------------------------------
	# Kernel structure encoding for model saving and loading
	# --------------------------------------------------------

	def produce_base_kern_name(self, base_kernel):
	    kernel_name_mapping = {
	        'RBF': 'rbf',
	        'StdPeriodic': 'periodic',
	        'Linear': 'linear',
	        'Bias': 'constant',
	        'White': 'white'
	    }
	    base_kernel_type = type(base_kernel).__name__
	    return kernel_name_mapping[base_kernel_type]

	def produce_kern_structure(self, kernel):
	    if isinstance(kernel, Add):
	        return {
	            'type': 'add',
	            'params': [self.produce_kern_structure(component_kern) for component_kern in kernel.parameters]
	        }
	    elif isinstance(kernel, Prod):
	        return {
	            'type': 'mult',
	            'params': [self.produce_kern_structure(component_kern) for component_kern in kernel.parameters]
	        }

	    return {
	        'type': self.produce_base_kern_name(kernel),
	        'params': kernel.param_array.tolist(),
	    }

	def parse_base_kern(self, kern_name, kern_params):
		# Necessary to include the creation methods so that kernel parameters are connected to their respective priors.
	    kernel_classes_mapping = {
	        'rbf': self.create_rbf_kernel,
	        'periodic': self.create_periodic_kernel,
	        'linear': self.create_linear_kernel,
	        'constant': self.create_constant_kernel,
	        'white': self.create_white_kernel
	    }
	    kern = kernel_classes_mapping[kern_name]()
	    kern[:] = kern_params
	    return kern

	def parse_kern_from_structure(self, kern_structure):
	    if kern_structure['type'] == 'add':
	        return Add([self.parse_kern_from_structure(component_kern) for component_kern in kern_structure['params']])
	    elif kern_structure['type'] == 'mult':
	        return Prod([self.parse_kern_from_structure(component_kern) for component_kern in kern_structure['params']])
	    return self.parse_base_kern(kern_structure['type'], kern_structure['params'])

	# ---------------------------------------------
	# Overrides for more general kernel management
	# ---------------------------------------------

	def generate_priors(self):
		super(KSiMGPE, self).generate_priors()
		
		# Periodic kernel
		self.priors['periodic_variance'] = GPy.priors.InverseGamma(1. + 0.1 * np.random.random(), 1.)
		self.update_inverse_gamma_params(self.priors['periodic_variance'], a=1.)

		self.priors['periodic_period'] = GPy.priors.InverseGamma(1. + 0.1 * np.random.random(), 1.)
		self.update_inverse_gamma_params(self.priors['periodic_period'], a=1.)

		self.priors['periodic_lengthscale'] = GPy.priors.LogGaussian(1. + 0.1 * np.random.random(), 1.)
		self.priors['periodic_lengthscale'].mu = 1.
		
		# Linear kernel
		self.priors['linear_variance'] = GPy.priors.InverseGamma(1. + 0.1 * np.random.random(), 1.)
		self.update_inverse_gamma_params(self.priors['linear_variance'], a=1.)

		# Constant kernel
		self.priors['constant_variance'] = GPy.priors.InverseGamma(1. + 0.1 * np.random.random(), 1.)
		self.update_inverse_gamma_params(self.priors['constant_variance'], a=1.)

	def create_base_kernel(self):
		base_kernels = ['rbf', 'periodic', 'linear', 'constant', 'white']
		picked_kernel = np.random.choice(base_kernels)
		
		if picked_kernel == 'rbf':
			return self.create_rbf_kernel()
		elif picked_kernel == 'periodic':
			return self.create_periodic_kernel()
		elif picked_kernel == 'linear':
			return self.create_linear_kernel()
		elif picked_kernel == 'constant':
			return self.create_constant_kernel()
		
		assert picked_kernel == 'white'
		return self.create_white_kernel()

	def save_model(self, filename='gimgpe_model.txt', folder=None):
		if filename is True:
			filename = "gimgpe_model_%s.txt" % datetime.datetime.now().strftime("%d%m%y%H%M%S")

		if folder is not None:
			filename = "%s/%s" % (folder, filename)

		model_state = {
			'c': self.c.tolist(),
			'alpha': self.alpha,
			'gating_phi': self.gating_phi,
			'kernels': [self.produce_kern_structure(kern) for kern in self.kernels],
			'prior_params': self.extract_gamma_prior_states()
		}

		with open(filename, 'w') as f:
			json.dump(model_state, f)

	def load_model(self, filename, folder=None):
		if folder is not None:
			filename = "%s/%s" % (folder, filename)

		with open(filename) as f:
			model_state = json.load(f)

		self.c = np.array(model_state['c'])
		self.alpha = model_state['alpha']
		self.gating_phi = model_state['gating_phi']
		self.class_indices = [np.nonzero(self.c == j)[0] for j in range(len(set(self.c)))]

		self.kernels = [self.parse_kern_from_structure(kern_structure) for kern_structure in model_state['kernels']]
		for prior_name, prior_params in model_state['prior_params'].iteritems():
			self.update_inverse_gamma_params(self.priors[prior_name], a=prior_params[0], b=prior_params[1])

		self.compute_singletons()
		self.compute_gating_distances()
		self.compute_gating_distances_class()

	# --------------------
	# Kernel measurement
	# --------------------

	def mle(self, j, kernel=None):
		# This portion of the logic to allow optimizing a kernel against the given class' data
		# when said kernel is not already represented by the class i.e. when optimizing a replacement
		# kernel in the kernel search step.
		if kernel is None:
			kernel = self.kernels[j]

		param_array = kernel.param_array.astype(np.float64)
		data_indices = self.class_indices[j]
		m = GPy.models.GPRegression(self.x[data_indices], self.y[data_indices], kernel, noise_var=0.0)
		m.likelihood.variance.fix(warning=False)
		try:
			m.optimize()
		except:
			kernel[:] = param_array # In case of numerical issues
		m.unlink_parameter(kernel)

	def measure_bic(self, j, kernel):
		data_indices = self.class_indices[j]
		class_point_count = len(data_indices)
		parameter_count = len(kernel.param_array)
		GP = GPy.models.GPRegression(self.x[data_indices], self.y[data_indices], kernel, noise_var=0.0)
		bic = -2 * GP.log_likelihood() + parameter_count * math.log(class_point_count)
		GP.unlink_parameter(kernel)
		return bic

	# -----------------------------------------
	# Chain iteration with kernel replacement
	# -----------------------------------------

	def iterate_chain(self, iterations=1, save_output=False, output_folder='.'):
		# Other variants can implement the necessary MCMC learning steps, but this class by itself
		# does not specify the kernel replacement method, and so it doesn't make sense to learn
		# from this class.
		pass