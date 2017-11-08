from base_test import BaseTest
import unittest
import numpy as np
from context import iMGPE

class ARSUnitTests(BaseTest):
	def test_singleton_alpha_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		pre_alpha = imgpe.alpha
		imgpe.ars_sample_alpha()
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)
		post_alpha = imgpe.alpha

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)
		self.assertEqual(pre_priors, post_priors)

		self.assertNotEqual(pre_alpha, post_alpha)

	def test_non_singleton_alpha_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		pre_alpha = imgpe.alpha
		imgpe.ars_sample_alpha()
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)
		post_alpha = imgpe.alpha

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)
		self.assertEqual(pre_priors, post_priors)

		self.assertNotEqual(pre_alpha, post_alpha)

	def test_prior_prime(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=29)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		for log_alpha in [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]:
			analytic_1st = imgpe.log_alpha_posterior_prime(log_alpha)
			upper_prior = imgpe.log_alpha_posterior(log_alpha + TINY)
			lower_prior = imgpe.log_alpha_posterior(log_alpha - TINY)
			central_diff_1st = (upper_prior - lower_prior) / (2 * TINY)
			self.assertAlmostEqual(analytic_1st, central_diff_1st, places=5)

if __name__ == '__main__':
	unittest.main()