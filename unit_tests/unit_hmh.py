from base_test import BaseTest
import unittest
import numpy as np
from context import iMGPE

class HMHUnitTests(BaseTest):
	def test_singleton_phi_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6, initial_gating_phi=0.1)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		pre_phi = imgpe.gating_phi

		np.random.seed(1008)
		imgpe.hmh_gating_phi()

		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)
		post_phi = imgpe.gating_phi

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)
		self.assertEqual(pre_priors, post_priors)

		self.assertNotEqual(pre_phi, post_phi)

	def test_non_singleton_phi_modified(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		pre_priors = BaseTest.extract_prior_state(imgpe)
		pre_phi = imgpe.gating_phi

		np.random.seed(1008)
		imgpe.hmh_gating_phi()

		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)
		post_priors = BaseTest.extract_prior_state(imgpe)
		post_phi = imgpe.gating_phi

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)
		self.assertEqual(pre_priors, post_priors)

		self.assertNotEqual(pre_phi, post_phi)

	def test_singleton_first_derivatives_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		epsilon = 1.25e-7
		old_phi = imgpe.gating_phi
		analytic_1st = imgpe.gating_log_distances_1st
		analytic_1st_to_rest = imgpe.gating_log_distances_1st_to_rest
		analytic_1st_to_class = imgpe.gating_log_distances_own_class_1st
		analytic_1st_ll = imgpe.pseudo_ll_1st

		imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		lower_gating_distances = imgpe.gating_distances
		lower_gating_distances_to_rest = imgpe.gating_distances_to_rest
		lower_gating_distances_to_class = imgpe.gating_distances_own_class
		lower_ll = imgpe.pseudo_ll

		imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		upper_gating_distances = imgpe.gating_distances
		upper_gating_distances_to_rest = imgpe.gating_distances_to_rest
		upper_gating_distances_to_class = imgpe.gating_distances_own_class
		upper_ll = imgpe.pseudo_ll

		central_diff_1st = (upper_gating_distances - lower_gating_distances) / (2 * TINY)
		central_diff_1st_to_rest = (upper_gating_distances_to_rest - lower_gating_distances_to_rest) / (2 * TINY)
		central_diff_1st_to_class = (upper_gating_distances_to_class - lower_gating_distances_to_class) / (2 * TINY)
		central_diff_1st_ll = (upper_ll - lower_ll) / (2 * TINY)
		
		self.assertTrue(np.allclose(analytic_1st, central_diff_1st, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_1st_to_rest, central_diff_1st_to_rest, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_1st_to_class, central_diff_1st_to_class, rtol=epsilon))
		self.assertAlmostEqual(analytic_1st_ll, central_diff_1st_ll, places=5)

	def test_non_singleton_first_derivatives_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		epsilon = 1.25e-7
		old_phi = imgpe.gating_phi
		analytic_1st = imgpe.gating_log_distances_1st
		analytic_1st_to_rest = imgpe.gating_log_distances_1st_to_rest
		analytic_1st_to_class = imgpe.gating_log_distances_own_class_1st
		analytic_1st_ll = imgpe.pseudo_ll_1st

		imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		lower_gating_distances = imgpe.gating_distances
		lower_gating_distances_to_rest = imgpe.gating_distances_to_rest
		lower_gating_distances_to_class = imgpe.gating_distances_own_class
		lower_ll = imgpe.pseudo_ll

		imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		upper_gating_distances = imgpe.gating_distances
		upper_gating_distances_to_rest = imgpe.gating_distances_to_rest
		upper_gating_distances_to_class = imgpe.gating_distances_own_class
		upper_ll = imgpe.pseudo_ll

		central_diff_1st = (upper_gating_distances - lower_gating_distances) / (2 * TINY)
		central_diff_1st_to_rest = (upper_gating_distances_to_rest - lower_gating_distances_to_rest) / (2 * TINY)
		central_diff_1st_to_class = (upper_gating_distances_to_class - lower_gating_distances_to_class) / (2 * TINY)
		central_diff_1st_ll = (upper_ll - lower_ll) / (2 * TINY)
		
		self.assertTrue(np.allclose(analytic_1st, central_diff_1st, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_1st_to_rest, central_diff_1st_to_rest, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_1st_to_class, central_diff_1st_to_class, rtol=epsilon))
		self.assertAlmostEqual(analytic_1st_ll, central_diff_1st_ll, places=5)

	def test_singleton_second_derivatives_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		epsilon = 1.25e-7
		old_phi = imgpe.gating_phi
		analytic_2nd = imgpe.gating_log_distances_2nd
		analytic_2nd_to_rest = imgpe.gating_log_distances_2nd_to_rest
		analytic_2nd_to_class = imgpe.gating_log_distances_own_class_2nd
		analytic_2nd_ll = imgpe.pseudo_ll_2nd

		imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		lower_gating_distances_1st = imgpe.gating_log_distances_1st
		lower_gating_distances_1st_to_rest = imgpe.gating_log_distances_1st_to_rest
		lower_gating_distances_1st_to_class = imgpe.gating_log_distances_own_class_1st
		lower_1st_ll = imgpe.pseudo_ll_1st

		imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		upper_gating_distances_1st = imgpe.gating_log_distances_1st
		upper_gating_distances_1st_to_rest = imgpe.gating_log_distances_1st_to_rest
		upper_gating_distances_1st_to_class = imgpe.gating_log_distances_own_class_1st
		upper_1st_ll = imgpe.pseudo_ll_1st

		central_diff_2nd = (upper_gating_distances_1st - lower_gating_distances_1st) / (2 * TINY)
		central_diff_2nd_to_rest = (upper_gating_distances_1st_to_rest - lower_gating_distances_1st_to_rest) / (2 * TINY)
		central_diff_2nd_to_class = (upper_gating_distances_1st_to_class - lower_gating_distances_1st_to_class) / (2 * TINY)
		central_diff_2nd_ll = (upper_1st_ll - lower_1st_ll) / (2 * TINY)
		
		self.assertTrue(np.allclose(analytic_2nd, central_diff_2nd, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_2nd_to_rest, central_diff_2nd_to_rest, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_2nd_to_class, central_diff_2nd_to_class, rtol=epsilon))
		self.assertAlmostEqual(analytic_2nd_ll, central_diff_2nd_ll, places=5)

	def test_non_singleton_second_derivatives_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		epsilon = 1.25e-7
		old_phi = imgpe.gating_phi
		analytic_2nd = imgpe.gating_log_distances_2nd
		analytic_2nd_to_rest = imgpe.gating_log_distances_2nd_to_rest
		analytic_2nd_to_class = imgpe.gating_log_distances_own_class_2nd
		analytic_2nd_ll = imgpe.pseudo_ll_2nd

		imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		lower_gating_distances_1st = imgpe.gating_log_distances_1st
		lower_gating_distances_1st_to_rest = imgpe.gating_log_distances_1st_to_rest
		lower_gating_distances_1st_to_class = imgpe.gating_log_distances_own_class_1st
		lower_1st_ll = imgpe.pseudo_ll_1st

		imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
		imgpe.compute_gating_distances()
		imgpe.compute_gating_distances_class()
		upper_gating_distances_1st = imgpe.gating_log_distances_1st
		upper_gating_distances_1st_to_rest = imgpe.gating_log_distances_1st_to_rest
		upper_gating_distances_1st_to_class = imgpe.gating_log_distances_own_class_1st
		upper_1st_ll = imgpe.pseudo_ll_1st

		central_diff_2nd = (upper_gating_distances_1st - lower_gating_distances_1st) / (2 * TINY)
		central_diff_2nd_to_rest = (upper_gating_distances_1st_to_rest - lower_gating_distances_1st_to_rest) / (2 * TINY)
		central_diff_2nd_to_class = (upper_gating_distances_1st_to_class - lower_gating_distances_1st_to_class) / (2 * TINY)
		central_diff_2nd_ll = (upper_1st_ll - lower_1st_ll) / (2 * TINY)
		
		self.assertTrue(np.allclose(analytic_2nd, central_diff_2nd, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_2nd_to_rest, central_diff_2nd_to_rest, rtol=epsilon))
		self.assertTrue(np.allclose(analytic_2nd_to_class, central_diff_2nd_to_class, rtol=epsilon))
		self.assertAlmostEqual(analytic_2nd_ll, central_diff_2nd_ll, places=5)

	def test_singleton_reject_kernels_unchanged(self):
		np.random.seed(1008)

		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6, initial_gating_phi=5.0)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		pre_log_distances = imgpe.gating_distances.copy()
		pre_log_distances_1st = imgpe.gating_log_distances_1st.copy()
		pre_log_distances_2nd = imgpe.gating_log_distances_2nd.copy()
		pre_log_distances_to_rest = imgpe.gating_distances_to_rest.copy()
		pre_log_distances_1st_to_rest = imgpe.gating_log_distances_1st_to_rest.copy()
		pre_log_distances_2nd_to_rest = imgpe.gating_log_distances_2nd_to_rest.copy()
		pre_log_distances_own_class = imgpe.gating_distances_own_class.copy()
		pre_log_distances_own_class_1st = imgpe.gating_log_distances_own_class_1st.copy()
		pre_log_distances_own_class_2nd = imgpe.gating_log_distances_own_class_2nd.copy()

		imgpe.hmh_gating_phi()

		post_log_distances = imgpe.gating_distances.copy()
		post_log_distances_1st = imgpe.gating_log_distances_1st.copy()
		post_log_distances_2nd = imgpe.gating_log_distances_2nd.copy()
		post_log_distances_to_rest = imgpe.gating_distances_to_rest.copy()
		post_log_distances_1st_to_rest = imgpe.gating_log_distances_1st_to_rest.copy()
		post_log_distances_2nd_to_rest = imgpe.gating_log_distances_2nd_to_rest.copy()
		post_log_distances_own_class = imgpe.gating_distances_own_class.copy()
		post_log_distances_own_class_1st = imgpe.gating_log_distances_own_class_1st.copy()
		post_log_distances_own_class_2nd = imgpe.gating_log_distances_own_class_2nd.copy()

		self.assertTrue(np.array_equal(pre_log_distances, post_log_distances))
		self.assertTrue(np.array_equal(pre_log_distances_1st, post_log_distances_1st))
		self.assertTrue(np.array_equal(pre_log_distances_2nd, post_log_distances_2nd))
		self.assertTrue(np.array_equal(pre_log_distances_to_rest, post_log_distances_to_rest))
		self.assertTrue(np.array_equal(pre_log_distances_1st_to_rest, post_log_distances_1st_to_rest))
		self.assertTrue(np.array_equal(pre_log_distances_2nd_to_rest, post_log_distances_2nd_to_rest))
		self.assertTrue(np.array_equal(pre_log_distances_own_class, post_log_distances_own_class))
		self.assertTrue(np.array_equal(pre_log_distances_own_class_1st, post_log_distances_own_class_1st))
		self.assertTrue(np.array_equal(pre_log_distances_own_class_2nd, post_log_distances_own_class_2nd))

	def test_prior_derivatives_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		mu = 0.
		var = 10000.
		for gating_phi in [0.5, 1, 2, 4]:
			imgpe.gating_phi = gating_phi

			old_phi = imgpe.gating_phi
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()

			analytic_1st = imgpe.log_prior_1st_deriv(mu, var)
			analytic_2nd = imgpe.log_prior_2nd_deriv(mu, var)

			imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()
			lower_prior = imgpe.log_prior(mu, var)
			lower_prior_1st = imgpe.log_prior_1st_deriv(mu, var)

			imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()
			upper_prior = imgpe.log_prior(mu, var)
			upper_prior_1st = imgpe.log_prior_1st_deriv(mu, var)

			central_diff_1st = (upper_prior - lower_prior) / (2 * TINY)
			central_diff_2nd = (upper_prior_1st - lower_prior_1st) / (2 * TINY)

			self.assertAlmostEqual(analytic_1st, central_diff_1st, places=5)
			self.assertAlmostEqual(analytic_2nd, central_diff_2nd, places=5)

	def test_non_singleton_fgh_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		for gating_phi in [0.5, 1, 2, 4]:
			imgpe.gating_phi = gating_phi

			old_phi = imgpe.gating_phi
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()

			_, analytic_g, analytic_h = imgpe.log_posterior_1st_2nd()

			imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()
			lower_f, lower_g, _ = imgpe.log_posterior_1st_2nd()

			imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()
			upper_f, upper_g, _ = imgpe.log_posterior_1st_2nd()

			central_diff_g = (upper_f - lower_f) / (2 * TINY)
			central_diff_h = (upper_g - lower_g) / (2 * TINY)

			self.assertAlmostEqual(analytic_g, central_diff_g, places=5)
			self.assertAlmostEqual(analytic_h, central_diff_h, places=5)

	def test_singleton_fgh_within_epsilon(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		TINY = 1.25e-7
		for gating_phi in [0.5, 1, 2, 4]:
			imgpe.gating_phi = gating_phi

			old_phi = imgpe.gating_phi
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()

			_, analytic_g, analytic_h = imgpe.log_posterior_1st_2nd()

			imgpe.gating_phi = np.exp(np.log(old_phi) - TINY)
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()
			lower_f, lower_g, _ = imgpe.log_posterior_1st_2nd()

			imgpe.gating_phi = np.exp(np.log(old_phi) + TINY)
			imgpe.compute_gating_distances()
			imgpe.compute_gating_distances_class()
			upper_f, upper_g, _ = imgpe.log_posterior_1st_2nd()

			central_diff_g = (upper_f - lower_f) / (2 * TINY)
			central_diff_h = (upper_g - lower_g) / (2 * TINY)

			self.assertAlmostEqual(analytic_g, central_diff_g, places=5)
			self.assertAlmostEqual(analytic_h, central_diff_h, places=5)

	def test_singleton_kern_to_class_match(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)

		distances_own_class = imgpe.gating_distances_own_class.copy()
		distances_own_class_recompute = []
		
		for i, (x, c) in enumerate(zip(imgpe.x, imgpe.c)):
			distance_own_class_point = sum([imgpe.gating_kern(x, x_other) for i_other, (x_other, c_other) in enumerate(zip(imgpe.x, imgpe.c)) if i_other != i and c_other == c])
			distances_own_class_recompute.append(distance_own_class_point)

		self.assertTrue(np.allclose(distances_own_class, distances_own_class_recompute))

if __name__ == '__main__':
	unittest.main()