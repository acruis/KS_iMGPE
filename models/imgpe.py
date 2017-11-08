import numpy as np
import GPy
import math
import itertools
import json
import random
import datetime
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.misc import logsumexp
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import gammaln, digamma

class iMGPE(object):
	"""
	The following is the full state of the model:

	x (ndarray) (n, 1)
	y (ndarray) (n, 1)
	c (ndarray) (n,)
	class_indices (list of ndarrays) (k,), where k is the number of classes
	priors (list): [kernel_type]_[kernel_hyper]_prior
	kernels (list) (k,)
	alpha
	gating_phi
	gating_distances (ndarray)
	gating_distances_own_class (ndarray)
	gating_distances_to_rest (ndarray)
	gating_log_distances_[1st/2nd] (ndarray) <--- somewhat of a misnomer. df/dlog(phi), not dlog(phi)/dphi
	gating_log_distances_to_rest_[1st/2nd] (ndarray) <--- same, df/dlog(phi)
	gating_log_distances_[1st/2nd] (ndarray) <--- same, df/dlog(phi)
	pseudo_ll (scalar) & 1st & 2nd <--- for HMH
	"""

	def __init__(self, x, y, initial_k=None, initial_alpha=2.15, initial_gating_phi=0.5):
		self.x = x
		self.y = y
		
		self.c, self.class_indices = self.partition_input(initial_k)
		
		self.generate_priors()
		self.kernels = self.generate_kernels()
		
		self.alpha = initial_alpha

		self.gating_phi = initial_gating_phi
		self.compute_singletons()
		self.compute_gating_distances()
		self.compute_gating_distances_class()

	# --------------------------
	# Kernel management system
	# --------------------------

	def create_rbf_kernel(self):
		kern = GPy.kern.RBF(input_dim=1, variance=self.priors['rbf_variance'].rvs(1), lengthscale=self.priors['rbf_variance'].rvs(1))
		kern.variance.set_prior(self.priors['rbf_variance'], warning=False)
		kern.lengthscale.set_prior(self.priors['rbf_lengthscale'], warning=False)
		return kern

	def create_white_kernel(self):
		kern = GPy.kern.White(input_dim=1., variance=self.priors['white_variance'].rvs(1))
		kern.variance.set_prior(self.priors['white_variance'], warning=False)
		return kern

	def create_base_kernel(self):
		return self.create_rbf_kernel() + self.create_white_kernel()

	def create_new_kernel(self):
		return self.create_base_kernel()

	# ---------------
	# Gating system
	# ---------------

	def gating_kern(self, x_1, x_2):
		return math.exp(-0.5 * ((x_1 - x_2)**2) / (self.gating_phi**2))

	def gating_log_kern_1st_incomplete(self, x_1, x_2):
		"""
		The full 1st-derivative with respect to the log gating_phi of the gating kernel is:

		[((x_1 - x_2) ** 2) / (self.gating_phi**2)] * self.gating_kern(x_1, x_2)
		"""
		return ((x_1 - x_2) ** 2) / (self.gating_phi**2)

	def gating_log_kern_2nd_incomplete(self, x_1, x_2):
		"""
		The full 2nd-derivative with respect to the log gating_phi of the gating kernel is:

		(((-2 * (self.gating_phi**2)) * ((x_1 - x_2) ** 2))
					+ ((x_1 - x_2) ** 4))
				/ (self.gating_phi**4) * self.gating_kern(x_1, x_2)
		"""
		return (((-2 * (self.gating_phi**2)) * ((x_1 - x_2) ** 2)) \
					+ ((x_1 - x_2) ** 4)) \
				/ (self.gating_phi**4)

	def class_mask(self):
		"""
		Generates an array that, when multiplied by a symmetric n-by-n distance array,
		produces an array where distances between points from different classes are 
		masked out (i.e. multiplied by 0).
		"""
		k = len(self.class_indices)
		masks = np.empty([k, len(self.c)])
		for j in range(k):
			masks[j] = np.zeros_like(self.c)
			masks[j][np.nonzero(self.c == j)[0]] = 1
		mask = np.empty([len(self.c), len(self.c)])
		for i in range(len(self.c)):
			mask[i] = masks[self.c[i]]
		return mask

	def compute_singletons(self):
		"""
		This method discovers data points which do not share a class with any other point.
		Used together with mask_singletons_gating_class to avoid division-by-zero.
		"""
		self.singletons = [class_indices_j[0] for class_indices_j in self.class_indices if len(class_indices_j) == 1]

	def mask_singletons_gating_class(self):
		self.gating_distances_own_class[self.singletons] = np.ma.masked
		self.gating_log_distances_own_class_1st[self.singletons] = np.ma.masked
		self.gating_log_distances_own_class_2nd[self.singletons] = np.ma.masked

	def compute_gating_distances(self):
		"""
		We compute the gating distance whenever the gating_phi changes.
		"""
		# For Gibbs
		gating_distances_evals = pdist(self.x, metric=self.gating_kern)
		self.gating_distances = squareform(gating_distances_evals)
		self.gating_distances_to_rest = np.sum(self.gating_distances, axis=1)

		# For HMH
		gating_log_distances_1st_incomplete = pdist(self.x, metric=self.gating_log_kern_1st_incomplete)
		gating_log_distances_2nd_incomplete = pdist(self.x, metric=self.gating_log_kern_2nd_incomplete)
		
		gating_log_distances_1st_evals = gating_log_distances_1st_incomplete * gating_distances_evals
		gating_log_distances_2nd_evals = gating_log_distances_2nd_incomplete * gating_distances_evals
		
		self.gating_log_distances_1st = squareform(gating_log_distances_1st_evals)
		self.gating_log_distances_2nd = squareform(gating_log_distances_2nd_evals)
		self.gating_log_distances_1st_to_rest = np.sum(self.gating_log_distances_1st, axis=1)
		self.gating_log_distances_2nd_to_rest = np.sum(self.gating_log_distances_2nd, axis=1)

	def log_pseudo_likelihood_kernterms(self):
		# OK to get a warning here as phi_prop might be small, causing some data points' distance to be 0 in gating_log_distances_to_rest.
		return np.ma.log(self.gating_distances_own_class) - np.log(self.gating_distances_to_rest)

	def log_pseudo_likelihood_n_alpha_term(self):
		n = len(self.c)
		return n * (np.log(n - 1) - np.log(n - 1 + self.alpha))

	def log_pseudo_likelihood(self):
		"""
		We produce ((n - 1)/(n - 1 + alpha)) * (sum all gating distances to class / sum all gating distances to all other data points)
		here, which are the modified conditionals in the iMGPE paper.
		"""
		return self.log_pseudo_likelihood_kernterms().sum() \
			 + self.log_pseudo_likelihood_n_alpha_term()

	def log_pseudo_likelihood_1st_deriv(self):
		return ((self.gating_log_distances_own_class_1st / self.gating_distances_own_class) - (self.gating_log_distances_1st_to_rest / self.gating_distances_to_rest)).sum()

	def log_pseudo_likelihood_2nd_deriv(self):
		return (((self.gating_log_distances_own_class_2nd / self.gating_distances_own_class) - ((self.gating_log_distances_own_class_1st / self.gating_distances_own_class) ** 2)) \
			 - ((self.gating_log_distances_2nd_to_rest / self.gating_distances_to_rest) - ((self.gating_log_distances_1st_to_rest / self.gating_distances_to_rest) ** 2))).sum()

	def compute_gating_distances_class(self): # For HMH
		mask = self.class_mask()
		self.gating_distances_own_class = np.ma.array(np.sum(self.gating_distances * mask, axis=1))
		self.gating_log_distances_own_class_1st = np.ma.array(np.sum(self.gating_log_distances_1st * mask, axis=1))
		self.gating_log_distances_own_class_2nd = np.ma.array(np.sum(self.gating_log_distances_2nd * mask, axis=1))

		self.mask_singletons_gating_class() # Necessary to avoid producing nan or -inf for singleton class.
		self.pseudo_ll = self.log_pseudo_likelihood()
		self.pseudo_ll_1st = self.log_pseudo_likelihood_1st_deriv()
		self.pseudo_ll_2nd = self.log_pseudo_likelihood_2nd_deriv()

	# -------
	# Gibbs
	# -------

	def ll_to_cluster(self, i):
		def ll_to_cluster_given_i(j):
			data_indices = [iprime for iprime in self.class_indices[j] if not iprime == i]
			# This is the bottleneck of the entire chain. iMGPE does this with rank-one updates,
			# but we are stuck with n^3 matrix inversion, because GPy does not support incrementally
			# adding new data points to the same GP.
			GP = GPy.models.GPRegression(self.x[data_indices], self.y[data_indices], self.kernels[j], noise_var=0.0)
			ll = np.asscalar(GP.log_predictive_density(np.array([self.x[i]]), np.array([self.y[i]])))
			GP.unlink_parameter(self.kernels[j]) # Necessary to unlink afterwards
			return ll if not np.isnan(ll) else -np.Inf # In case of large variance
		return ll_to_cluster_given_i

	def sum_kern_to_class(self, i):
		def sum_kern_to_single_class(j):
			return self.gating_distances[i][self.class_indices[j]].sum()
		return sum_kern_to_single_class

	def log_indicator_conditionals(self, kern_to_classes, i):
		n = len(self.x)
		return math.log(n - 1) - math.log(n - 1 + self.alpha) + np.log(kern_to_classes) - np.log(self.gating_distances_to_rest[i]) # Fine to get warning here due to kern_to_class especially if phi is small.

	def log_posterior_auxiliary_class(self, i, kernel):
		n = len(self.x)
		log_conditional_to_new_class = math.log(self.alpha) - math.log(n - 1 + self.alpha)
		# Mean at zero, and variance equal to applying the kernel between the input coordinate and itself
		# When the old class is too terrible and norm.pdf just gives 0, we need np.log to produce -inf so that we can avoid numerical issues in computing logsumexp and the cumsum later.
		log_likelihood_to_new_class = np.log(np.asscalar(norm.pdf(self.y[i], 0, kernel.K(np.array([self.x[i]]))))) # Only pass one argument to K to make sure kernels that behave differently for prediction (ie White) works correctly.
		return log_conditional_to_new_class + log_likelihood_to_new_class

	def log_posterior_own_class(self, compute_log_likelihood_to_cluster, compute_kern_to_classes, i, j):
		if len(self.class_indices[j]) > 1:
			# Like any other cluster
			return compute_log_likelihood_to_cluster(j) + self.log_indicator_conditionals(compute_kern_to_classes(j), i)
		else:
			# Singleton case
			return self.log_posterior_auxiliary_class(i, self.kernels[j])

	def move_point_to_class(self, i, sample, new_class_created=False):
		old_class = self.c[i]

		if old_class == sample:
			return

		if new_class_created:
			self.class_indices.append(np.empty(0, dtype='int'))

		# Update state.
		self.class_indices[old_class] = self.class_indices[old_class][self.class_indices[old_class] != i]
		self.c[i] = sample
		self.class_indices[sample] = np.append(self.class_indices[sample], i)

		# Cleanup old class if it's no longer active.
		if len(self.class_indices[old_class]) < 1:
			self.class_indices.pop(old_class)
			self.kernels.pop(old_class)
			self.c[self.c > old_class] = self.c[self.c > old_class] - 1

	def gibbs(self, i):
		k = len(self.class_indices)
		n = len(self.x)

		# Setup computation for other clusters
		other_clusters = np.arange(k)
		other_clusters = other_clusters[other_clusters != self.c[i]]
		log_class_posteriors = np.zeros(k + 1)

		# F(y_i,c_i)
		compute_log_likelihood_to_cluster = np.vectorize(self.ll_to_cluster(i), otypes=[np.float64])
		log_likelihood_to_clusters = compute_log_likelihood_to_cluster(other_clusters)

		# n_-i,c / n - 1 + alpha
		compute_kern_to_classes = np.vectorize(self.sum_kern_to_class(i), otypes=[np.float64])
		kern_to_classes = compute_kern_to_classes(other_clusters)
		log_conditional_to_classes = self.log_indicator_conditionals(kern_to_classes, i)

		# Posterior for other clusters
		log_class_posteriors[other_clusters] = log_conditional_to_classes + log_likelihood_to_clusters
		# Posterior for own cluster
		log_class_posteriors[self.c[i]] = self.log_posterior_own_class(compute_log_likelihood_to_cluster, compute_kern_to_classes, i, self.c[i])

		# Create new kernel
		create_auxiliary_kernel = len(self.class_indices[self.c[i]]) > 1
		new_kernel = self.create_new_kernel() if create_auxiliary_kernel else None
		if create_auxiliary_kernel:
			log_class_posteriors[-1] = self.log_posterior_auxiliary_class(i, new_kernel)
		else:
			log_class_posteriors = log_class_posteriors[:-1]

		# Sample new class
		log_normalizer = logsumexp(log_class_posteriors)
		log_class_posteriors_scaled = log_class_posteriors - log_normalizer
		class_posteriors_scaled = np.exp(log_class_posteriors_scaled)
		class_posteriors_cumulative = np.cumsum(class_posteriors_scaled)
		u = np.random.random()
		sample = np.nonzero(class_posteriors_cumulative >= u)[0][0]

		# Move new point
		new_class_created = (sample >= k)
		self.move_point_to_class(i, sample, new_class_created)
		if new_class_created:
			self.kernels.append(new_kernel)

	def gibbs_round(self):
		for i in range(len(self.x)):
			self.gibbs(i)
		self.compute_singletons()
		# After performing a Gibbs round, the memberships have changed and so we have to update the relevant
		# state to prepare for HMH.
		self.compute_gating_distances_class() 

	# -----
	# HMC
	# -----

	def hmc(self, j):
		data_indices = self.class_indices[j]
		
		param_array = self.kernels[j].param_array.astype(np.float64)
		
		m = GPy.models.GPRegression(self.x[data_indices], self.y[data_indices], self.kernels[j], noise_var=0.0)
		m.likelihood.variance.fix(warning=False)
		hmc = GPy.inference.mcmc.HMC(m, stepsize=5e-4)
		
		try:
			_ = hmc.sample(num_samples=1, hmc_iters=10)
		except np.linalg.linalg.LinAlgError:
			self.kernels[j][:] = param_array # Upon numerical error, treat the step as a rejection.
		
		m.unlink_parameter(self.kernels[j])

	def hmc_round(self):
		for j in range(len(self.class_indices)):
			self.hmc(j)

	# -------------------------
	# Hyperhyper optimization
	# -------------------------

	def collect_gamma(self):
		# Returns a list of tuples, where each tuple is a prior and the list of hyperparameter values connected to it.
		priors_and_values = list(itertools.chain(*[[(prior, np.asscalar(kernel.param_array[hyperparam_index][0])) for prior, hyperparam_index in kernel.priors.items() if isinstance(prior, GPy.priors.InverseGamma)] for kernel in self.kernels]))
		priors_and_values.sort(key=lambda x: x[0]) # Necessary to sort for groupby to work properly.
		# groupby takes in (prior, value) tuples and groups them, but after they are grouped we only want the values within them - this is why we only take value from prior_and_value_tuples.
		grouped_inverse_gamma = [(prior, [value for prior, value in prior_and_value_tuples]) for prior, prior_and_value_tuples in itertools.groupby(priors_and_values, key=lambda x: x[0])]
		return grouped_inverse_gamma

	def update_inverse_gamma_params(self, gamma_prior, a=None, b=None):
		"""
		Deals with GPy priors exerting silly behaviour, where the internal (normalizing) constant
		does not get updated when the hyperparameters get updated.
		"""
		if a is not None:
			gamma_prior.a = a
		if b is not None:
			gamma_prior.b = b
		gamma_prior.constant = -gammaln(gamma_prior.a) + gamma_prior.a * math.log(gamma_prior.b)

	def inverse_gamma_lnpdf_grads(self, prior, x):
		alpha_grad = math.log(prior.b) - digamma(prior.a) - math.log(x)
		beta_grad = (prior.a / prior.b) - (1. / x)
		return (alpha_grad, beta_grad)

	def optimize_inverse_gamma_prior(self, prior, values):
		log_likelihood = np.vectorize(prior.lnpdf)
		def objective_and_grad(hypers):
			self.update_inverse_gamma_params(prior, a=hypers[0], b=hypers[1])
			log_likelihood_grads = [self.inverse_gamma_lnpdf_grads(prior, value) for value in values]
			alpha_grad_sum, beta_grad_sum = reduce(lambda (alpha_grad_1, beta_grad_1), (alpha_grad_2, beta_grad_2): (alpha_grad_1 + alpha_grad_2, beta_grad_1 + beta_grad_2), log_likelihood_grads)
			return -log_likelihood(values).sum(), np.array([-alpha_grad_sum, -beta_grad_sum])
		# We use a bounded minimize so that our prior.a and prior.b do not go to negative.
		minimize(objective_and_grad, np.array([prior.a, prior.b]), jac=True, bounds=((1e-14, None), (1e-14, None)))

	def optimize_priors(self):
		# We follow in iMGPE's footsteps and leave the lengthscale prior alone.
		grouped_inverse_gamma = self.collect_gamma()
		for prior, values in grouped_inverse_gamma:
			self.optimize_inverse_gamma_prior(prior, values)

	# ---------------
	# ARS for alpha
	# ---------------

	def log_alpha_posterior(self, y):
		k = len(self.class_indices)
		n = len(self.c)
		return ((k - 0.5) * y) \
				- (1/(2*math.exp(y))) \
				+ gammaln(math.exp(y)) \
				- gammaln(n + math.exp(y))

	def log_alpha_posterior_prime(self, y):
		# Also a misnomer. We are taking d/dlog(alpha), not dlog(alpha)/dalpha.
		k = len(self.class_indices)
		n = len(self.c)
		return (k - 0.5) \
				+ (0.5 * math.exp(-y)) \
				+ math.exp(y) * digamma(math.exp(y)) \
				- math.exp(y) * digamma(n + math.exp(y))

	def init_ars(self, abscissae, h, hprime, lb=-np.Inf, ub=np.Inf):
		h_x = h(abscissae)
		hprime_x = hprime(abscissae)
		x_hprime_x = abscissae * hprime_x
		
		h_diff = np.diff(h_x)
		hprime_diff = np.diff(hprime_x)
		x_hprime_diff = np.diff(x_hprime_x)

		intersects = np.zeros(len(abscissae) + 1)
		intersects[1:-1] = (h_diff - x_hprime_diff) / (-hprime_diff)
		intersects[0] = lb; intersects[-1] = ub

		k = len(abscissae) # = len(h_x) = len(hprime_x)
		u_intersects = h_x[[0] + range(k)] \
						+ (hprime_x[[0] + range(k)] \
							* (intersects - abscissae[[0] + range(k)]))
		S_x = np.hstack([0, np.cumsum(np.diff(np.exp(u_intersects))/hprime_x)])
		return intersects, u_intersects, S_x, h_x, hprime_x

	def sample_hull(self, abscissae, S_x, h_x, hprime_x, u_intersects):
		u = random.random()
		i = np.nonzero((S_x / S_x[-1]) < u)[0][-1]
		x_t = abscissae[i] \
			+ ((math.log( \
					math.exp(u_intersects[i]) + hprime_x[i] * (S_x[-1] * u - S_x[i])
				) - h_x[i]) / hprime_x[i])
		return x_t, i

	def rejection_test(self, abscissae, x_t, i, h, hprime, h_x, hprime_x):
		# Returns True if value NOT rejected.
		h_x_t = h(x_t)
		u_x_t = hprime_x[i] * (x_t - abscissae[i]) + h_x[i]
		w = random.random()
		return w <= math.exp(h_x_t - u_x_t)

	def ars_sample_alpha(self):
		# Covers a good range for alpha, as we are operating on the log scale.
		abscissae = np.array([-5., -4., -3., -2., -1., -0.5, 0, 0.5, 1, 2, 3, 4, 5])
		h = np.vectorize(self.log_alpha_posterior)
		hprime = np.vectorize(self.log_alpha_posterior_prime)

		intersects, u_intersects, S_x, h_x, hprime_x = self.init_ars(abscissae, h, hprime, lb=-6., ub=6.)
		x_t, i = self.sample_hull(abscissae, S_x, h_x, hprime_x, u_intersects)
		while not self.rejection_test(abscissae, x_t, i, h, hprime, h_x, hprime_x):
			x_t, i = self.sample_hull(abscissae, S_x, h_x, hprime_x, u_intersects)
		self.alpha = math.exp(x_t)

	# ----------------------------
	# HMH for gating lengthscale
	# ----------------------------

	def log_prior(self, mu, var):
		return -0.5 * math.log(2 * math.pi * var) - math.log(self.gating_phi) - (((math.log(self.gating_phi) - mu) ** 2) / (2 * var))

	def log_prior_1st_deriv(self, mu, var):
		return (-1) - ((math.log(self.gating_phi) - mu) / var)

	def log_prior_2nd_deriv(self, mu, var):
		return -1 / var

	def log_posterior_1st_2nd(self):
		mu = 0.
		var = 10000.
		log_prior_eval = self.log_prior(mu, var)
		log_prior_1st = self.log_prior_1st_deriv(mu, var)
		log_prior_2nd = self.log_prior_2nd_deriv(mu, var)

		f = log_prior_eval + self.pseudo_ll + math.log(self.gating_phi)
		g = log_prior_1st + self.pseudo_ll_1st + 1
		h = log_prior_2nd + self.pseudo_ll_2nd
		return f,g,h

	def sample_accept_reject(self):
		# The pseudo-posterior is not actually log-concave wrt log(gating_phi)
		# As such, sometimes the Hessian becomes positive...
		log_phi_old = math.log(self.gating_phi)
		f_old, g_old, h_old = self.log_posterior_1st_2nd()
		
		sigma_old = math.sqrt(abs(1/h_old))
		mu_old = log_phi_old + (abs(1/h_old) * g_old)

		if sigma_old > 2.0:
			# Or it reaches too close to zero, causing the proposal distribution to be too loose.
			# In which case we switch to Metropolis walk.
			sigma_old = 1.
			mu_old = log_phi_old

		log_phi_prop = np.random.normal(loc=mu_old, scale=sigma_old)
		q_forward = norm.logpdf(log_phi_prop, loc=mu_old, scale=sigma_old)

		self.gating_phi = math.exp(log_phi_prop)
		self.compute_gating_distances()
		self.compute_gating_distances_class()

		if np.any(self.gating_distances_to_rest == 0):
			# Proposed phi is too small, reject it so the Gibbs sampling step won't run into issues.
			return None

		f_prop, g_prop, h_prop = self.log_posterior_1st_2nd()

		sigma_prop = math.sqrt(abs(1/h_prop))
		mu_prop = log_phi_prop + (abs(1/h_prop) * g_prop)

		if sigma_prop > 2.0:
			sigma_prop = 1.
			mu_prop = log_phi_prop

		q_backward = norm.logpdf(log_phi_old, loc=mu_prop, scale=sigma_prop)
		r = math.exp((f_prop - f_old) + (q_backward - q_forward))

		return math.exp(log_phi_prop) if r > np.random.random() else None

	def hmh_gating_phi(self):
		old_gating_phi = self.gating_phi
		sample = self.sample_accept_reject()
		if sample is None:
			self.gating_phi = old_gating_phi
			self.compute_gating_distances()
			self.compute_gating_distances_class()
		else:
			self.gating_phi = sample

	# -------
	# Learn
	# -------

	def iterate_chain(self, iterations=1, skips=-1, save_output=False, output_folder='.'):
		for it in range(iterations):
			print it
			self.gibbs_round()
			self.hmc_round()
			self.optimize_priors()
			self.ars_sample_alpha()
			self.hmh_gating_phi()

			if save_output and (skips > 0) and (it % skips == 0):
				self.save_model(filename=("%d.txt" % it), folder=output_folder)

	# ---------
	# Predict
	# ---------

	def sample_point_assignment(self, x_star_i, x_x_star, c_c_star, k):
		n = len(x_x_star) + 1
		source = np.array([x_star_i])
		distances = cdist(x_x_star, source, metric=self.gating_kern)

		class_probabilities = []
		class_indices = [np.nonzero(c_c_star == j)[0] for j in range(k)]

		sum_dist_to_classes = [distances[class_indices[j]].sum() for j in range(k)]
		sum_dist_to_rest = distances.sum()

		conditionals_to_classes = [(n - 1) / (n - 1 + self.alpha) * sum_dist_to_class/sum_dist_to_rest for sum_dist_to_class in sum_dist_to_classes]
		conditionals_to_classes = [(0 if np.isnan(conditional_to_class) else conditional_to_class) for conditional_to_class in conditionals_to_classes]
		conditionals_to_classes.append(self.alpha / (n - 1 + self.alpha))
		conditionals_cumsum = np.cumsum(conditionals_to_classes)
		conditionals_cumsum_scaled = conditionals_cumsum / conditionals_cumsum[-1]
		u = np.random.random()
		target_class = np.nonzero(conditionals_cumsum_scaled >= u)[0][0]

		return target_class

	def predict_class(self, j, x_star):
		if len(x_star) == 0:
			return []

		data_indices = self.class_indices[j]
		GP = GPy.models.GPRegression(self.x[data_indices], self.y[data_indices], self.kernels[j], noise_var=0.0)
		posterior_samples = GP.posterior_samples(x_star, size=1)
		GP.unlink_parameter(self.kernels[j])
		return posterior_samples.flatten()

	def predict_new_class(self, x_star):
		mean_new = np.zeros(x_star.shape[0])

		kernel_new = self.create_new_kernel()
		cov_new = kernel_new.K(x_star)
		return np.random.multivariate_normal(mean_new, cov_new)

	def predict(self, x_star_row):
		x_star = np.array(np.matrix(x_star_row).T)

		x_x_star = self.x
		c_c_star = self.c
		c_star = []
		k = len(self.class_indices)

		for x_star_i in x_star:
			target_class = self.sample_point_assignment(x_star_i, x_x_star, c_c_star, k)
			
			# Optionally add x_star_i into x_x_star and target_class into c_c_star for non-independent samples.

			c_star.append(target_class)
			if target_class >= k:
				k = k + 1

		c_star = np.array(c_star)
		class_indices_star = [np.nonzero(c_star == j)[0] for j in range(k)]
		x_per_class = [x_star[class_indices_star_j] for class_indices_star_j in class_indices_star]

		k_with_data = len(self.class_indices)
		y_per_class = [self.predict_class(j, x_per_class_j) if j < len(self.class_indices) else self.predict_new_class(x_per_class_j) for j, x_per_class_j in enumerate(x_per_class)]

		y_star = np.zeros(x_star.shape[0])
		for class_indices_star_j, y_per_class_j in zip(class_indices_star, y_per_class):
			y_star[class_indices_star_j] = y_per_class_j
		
		return y_star

	def posterior(self):
		means = np.zeros_like(self.x)
		for j, data_indices in enumerate(self.class_indices):
			GP = GPy.models.GPRegression(self.x[data_indices], self.y[data_indices], self.kernels[j], noise_var=0.0)
			means[data_indices] = GP.predict(self.x[data_indices])[0]
			GP.unlink_parameter(self.kernels[j])
		return means

	# ------------------------
	# Model state management
	# ------------------------

	def extract_gamma_prior_states(self):
		gamma_prior_states = {}
		for prior_name, prior in self.priors.iteritems():
			if not isinstance(prior, GPy.priors.InverseGamma):
				continue
			gamma_prior_states[prior_name] = [prior.a, prior.b]
		return gamma_prior_states

	def save_model(self, filename='imgpe_model.txt', folder=None):
		if filename is True:
			filename = "imgpe_model_%s.txt" % datetime.datetime.now().strftime("%d%m%y%H%M%S")

		if folder is not None:
			filename = "%s/%s" % (folder, filename)

		model_state = {
			'c': self.c.tolist(),
			'alpha': self.alpha,
			'gating_phi': self.gating_phi,
			'kernel_params': [kernel.param_array.tolist() for kernel in self.kernels],
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

		self.kernels = self.generate_kernels() # Have to replace this with kernel structure reconstruction for KSiMGPE
		for kernel, kernel_params in zip(self.kernels, model_state['kernel_params']):
			kernel[:] = kernel_params

		for prior_name, prior_params in model_state['prior_params'].iteritems():
			self.update_inverse_gamma_params(self.priors[prior_name], a=prior_params[0], b=prior_params[1])

		self.compute_singletons()
		self.compute_gating_distances()
		self.compute_gating_distances_class()

	# --------------------------
	# Initial state generation
	# --------------------------

	def generate_priors(self):
		"""
		It might look silly that we're creating InverseGamma priors with random noise, and
		then immediately setting the parameters to 1,1.
		However, this is because the __new__ method for GPy priors is overridden to "enforce"
		that priors that get created with the same parameters must be the same instance (i.e. the
		singleton pattern).
		"""
		self.priors = {}
		
		# RBF kernel
		self.priors['rbf_variance'] = GPy.priors.InverseGamma(1. + 0.1 * np.random.random(), 1.)
		self.update_inverse_gamma_params(self.priors['rbf_variance'], a=1.)

		self.priors['rbf_lengthscale'] = GPy.priors.LogGaussian(1. + 0.1 * np.random.random(), 1.)
		self.priors['rbf_lengthscale'].mu = 1. # No need to do careful updating with a LogGaussian prior as mu is not part of the constant.

		# White kernel
		self.priors['white_variance'] = GPy.priors.InverseGamma(1. + 0.1 * np.random.random(), 1.)
		self.update_inverse_gamma_params(self.priors['white_variance'], a=1.)

	def generate_kernels(self):
		return [self.create_new_kernel() for _ in range(len(self.class_indices))]

	def partition_input(self, initial_k=None):
		"""
		Generate a random partition between data points upon initialization.
		"""
		k = int(initial_k) if initial_k is not None else min(len(self.x), 6)
		n = len(self.x)

		def generate_barriers():
			proportions = np.random.rand(k)
			proportions = np.cumsum(proportions / proportions.sum())
			barriers = proportions * n
			return barriers

		def empty_class_created(barriers):
			for i in range(1, k):
				if barriers[i] == barriers[i - 1]:
					return True

			return False

		barriers = generate_barriers()
		while (empty_class_created(barriers)):
			barriers = generate_barriers()

		c = np.ones(n)
		j = 0
		for i in range(n):
			if i > barriers[j]:
				j += 1
			c[i] = j
		return c.astype(int), [np.nonzero(c == j)[0] for j in range(k)]
