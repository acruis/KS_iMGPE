import unittest
import json
import numpy as np
import GPy

class BaseTest(unittest.TestCase):
	@staticmethod
	def get_data():
		with open('sample_data') as f:
			xyyerr_dict = json.load(f)

		return np.array(xyyerr_dict['x']), np.array(xyyerr_dict['y']), np.array(xyyerr_dict['yerr'])

	@staticmethod
	def divide_data(imgpe, singleton=False):
		if singleton:
			imgpe.c = np.concatenate([np.ones(22) * 0,
						  			  np.ones(21) * 1,
						  			  np.ones(1) * 2,
						  			  np.ones(15) * 3,
						  			  np.ones(20) * 4,
						  			  np.ones(21) * 5]).astype('int')
		else:
			imgpe.c = np.concatenate([np.ones(17) * 0,
									  np.ones(17) * 1,
									  np.ones(17) * 2,
									  np.ones(17) * 3,
									  np.ones(16) * 4,
									  np.ones(16) * 5]).astype('int')
		imgpe.class_indices = [np.nonzero(imgpe.c == j)[0] for j in range(6)]
		imgpe.compute_singletons()
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()

	@staticmethod
	def prepare_kernels(imgpe):
		for kernel in imgpe.kernels:
			kernel.rbf.variance = 1.
			kernel.rbf.lengthscale = 1.
			kernel.white.variance = 1.

	@staticmethod
	def boost_self_affinity(imgpe, i):
		self_j = imgpe.c[i]
		BaseTest.boost_target_affinity(imgpe, i, self_j, use_self=False)

	@staticmethod
	def boost_target_affinity(imgpe, i, target_cluster, use_self=True):
		k = len(imgpe.class_indices)
		for j in range(k):
			if j == target_cluster:
				data_indices = imgpe.class_indices[j] if (not use_self) else np.array([i]).astype('int')
				m = GPy.models.GPRegression(imgpe.x[data_indices], imgpe.y[data_indices], imgpe.kernels[j], noise_var=0.0)
				m.likelihood.variance.fix(warning=False)
				m.optimize()
				m.unlink_parameter(imgpe.kernels[j])
			else:
				imgpe.kernels[j].rbf.variance = 0.001
				imgpe.kernels[j].rbf.lengthscale = 20.
				imgpe.kernels[j].white.variance = 0.001

	"""
	The following two methods control whether Gibbs sampling will assign a point to a new class
	"""
	@staticmethod
	def destroy_priors(imgpe):
		imgpe.priors['rbf_variance'] = GPy.priors.InverseGamma(1001., 1.)
		imgpe.priors['rbf_lengthscale'] = GPy.priors.LogGaussian(3, 0.000001)
		imgpe.priors['white_variance'] = GPy.priors.InverseGamma(1001., 1.)

	@staticmethod
	def boost_priors(imgpe):
		imgpe.priors['rbf_variance'] = GPy.priors.InverseGamma(22910.750584579047, 109653.13761133047) # Produces 4.786 at variance 0.001
		imgpe.priors['rbf_lengthscale'] = GPy.priors.LogGaussian(0.4233406239265371, 0.02070169521990221) # Produces 1.527 at variance 0.001

	@staticmethod
	def extract_sample_state(imgpe):
		c = np.copy(imgpe.c)
		class_indices = []
		for indices_in_class in imgpe.class_indices:
			class_indices.append(np.copy(indices_in_class))
		param_vals = []
		for kern in imgpe.kernels:
			param_vals.append(kern.param_array.tolist())
		return c, class_indices, param_vals

	@staticmethod
	def extract_prior_state(imgpe):
		prior_state = {}
		for prior_name, prior in imgpe.priors.iteritems():
			if isinstance(prior, GPy.priors.InverseGamma):
				prior_state[prior_name] = (prior.a, prior.b)
			elif isinstance(prior, GPy.priors.LogGaussian):
				prior_state[prior_name] = (prior.mu, prior.sigma2)
		return prior_state