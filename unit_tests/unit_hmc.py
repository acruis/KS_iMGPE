from base_test import BaseTest
import unittest
import numpy as np
from context import iMGPE

class HMCUnitTests(BaseTest):
	def test_singleton_hyperparams_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		imgpe.hmc_round()
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(len(pre_param_vals), len(post_param_vals))
		for pre_param_vals_j, post_param_vals_j in zip(pre_param_vals, post_param_vals):
			self.assertEqual(len(pre_param_vals_j), len(post_param_vals_j))
			self.assertNotEqual(pre_param_vals_j, post_param_vals_j)

		self.assertEqual(pre_priors, post_priors)

	def test_non_singleton_hyperparams_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		imgpe.hmc_round()
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(len(pre_param_vals), len(post_param_vals))
		for pre_param_vals_j, post_param_vals_j in zip(pre_param_vals, post_param_vals):
			self.assertEqual(len(pre_param_vals_j), len(post_param_vals_j))
			self.assertNotEqual(pre_param_vals_j, post_param_vals_j)

		self.assertEqual(pre_priors, post_priors)

if __name__ == '__main__':
	unittest.main()