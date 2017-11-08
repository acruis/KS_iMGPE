from base_test import BaseTest
import unittest
import numpy as np
import GPy
from context import iMGPE

class HyperhyperOptimizeTests(BaseTest):
	@staticmethod
	def naive_prior_value_extraction(imgpe):
		priors_and_values = []
		for prior_name, prior in imgpe.priors.iteritems():
			if not isinstance(prior, GPy.priors.InverseGamma):
				continue
			values = []
			for kernel in imgpe.kernels:
				for param_prior, param_index in kernel.priors.items():
					if param_prior is prior:
						values.append(kernel.param_array[param_index][0])
			priors_and_values.append((prior, values))
		return priors_and_values

	def ensure_optimal_hyperhypers(self, gamma_prior, values):
		current_loglikelihood = sum([gamma_prior.lnpdf(value) for value in values])
		adjustments = [(-0.001, -0.001), (-0.001, 0.001), (0.001, -0.001), (0.001, 0.001)]
		for a_adjustment, b_adjustment in adjustments:
			prior = GPy.priors.InverseGamma(gamma_prior.a + a_adjustment, gamma_prior.b + b_adjustment)
			adjusted_loglikelihood = sum([prior.lnpdf(value) for value in values])
			self.assertTrue(adjusted_loglikelihood < current_loglikelihood)

	def test_collect_values_match(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)

		grouped_inverse_gamma = imgpe.collect_gamma()
		naive_grouped = self.naive_prior_value_extraction(imgpe)

		for prior, values in grouped_inverse_gamma:
			found_prior = False
			for naive_grouped_prior, naive_grouped_values in naive_grouped:
				if not (naive_grouped_prior is prior):
					continue
				found_prior = True
				self.assertItemsEqual(values, naive_grouped_values)
			self.assertTrue(found_prior)

	def test_singleton_hyperhypers_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		imgpe.optimize_priors()
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)

		self.assertNotEqual(pre_priors, post_priors)
		priors_and_values = self.naive_prior_value_extraction(imgpe)
		for prior, values in priors_and_values:
			self.ensure_optimal_hyperhypers(prior, values)

	def test_non_singleton_hyperhypers_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		imgpe.optimize_priors()
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)

		self.assertNotEqual(pre_priors, post_priors)
		priors_and_values = self.naive_prior_value_extraction(imgpe)
		for prior, values in priors_and_values:
			self.ensure_optimal_hyperhypers(prior, values)

if __name__ == '__main__':
	unittest.main()