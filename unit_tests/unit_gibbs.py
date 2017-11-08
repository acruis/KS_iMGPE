from base_test import BaseTest
import unittest
import numpy as np
from context import iMGPE

class GibbsUnitTests(BaseTest):
	def test_singleton_reassignment(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)
		BaseTest.boost_self_affinity(imgpe, 43)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		imgpe.gibbs(43)
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)

	def test_singleton_different_class(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=True)
		BaseTest.prepare_kernels(imgpe)
		target_cluster = 4
		BaseTest.boost_target_affinity(imgpe, 43, target_cluster)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		imgpe.gibbs(43)
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for index, (pre_c_i, post_c_i) in enumerate(zip(pre_c, post_c)):
			if index == 43:
				self.assertEqual(post_c_i, target_cluster - 1) # Shifted by one because target cluster is ahead
			else:
				if pre_c_i > 2:
					self.assertEqual(pre_c_i - 1, post_c_i)
				else:
					self.assertEqual(pre_c_i, post_c_i)

		pre_class_indices_compare = [pre_class_indices_j for index, pre_class_indices_j in enumerate(pre_class_indices) if index != 2 and index != target_cluster]
		post_class_indices_compare = [pre_class_indices_j for index, pre_class_indices_j in enumerate(post_class_indices) if index != target_cluster - 1]
		self.assertEqual(len(pre_class_indices) - 1, len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices_compare, post_class_indices_compare):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))
		self.assertTrue(np.array_equal(np.append(pre_class_indices[target_cluster], 43), post_class_indices[target_cluster - 1]))

		self.assertEqual(len(pre_param_vals) - 1, len(post_param_vals))
		pre_param_vals_compare = [pre_param_vals_j for index, pre_param_vals_j in enumerate(pre_param_vals) if index != 2]
		self.assertEqual(pre_param_vals_compare, post_param_vals)

	def test_nonsingleton_reassignment(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)
		BaseTest.boost_self_affinity(imgpe, 43)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		imgpe.gibbs(43)
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for pre_c_i, post_c_i in zip(pre_c, post_c):
			self.assertEqual(pre_c_i, post_c_i)

		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices, post_class_indices):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))

		self.assertEqual(pre_param_vals, post_param_vals)

	def test_nonsingleton_different_old_class(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)
		target_cluster = 4
		BaseTest.boost_target_affinity(imgpe, 43, target_cluster)
		BaseTest.destroy_priors(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		imgpe.gibbs(43)
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for index, (pre_c_i, post_c_i) in enumerate(zip(pre_c, post_c)):
			if index == 43:
				self.assertEqual(post_c_i, target_cluster)
			else:
				self.assertEqual(pre_c_i, post_c_i)

		pre_class_indices_compare = [pre_class_indices_j for index, pre_class_indices_j in enumerate(pre_class_indices) if index != 2 and index != target_cluster]
		post_class_indices_compare = [pre_class_indices_j for index, pre_class_indices_j in enumerate(post_class_indices) if index != 2 and index != target_cluster]
		self.assertEqual(len(pre_class_indices), len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices_compare, post_class_indices_compare):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))
		self.assertTrue(len(pre_class_indices[2]) - 1, len(post_class_indices[2]))
		self.assertTrue(np.array_equal(pre_class_indices[2][pre_class_indices[2] != 43], post_class_indices[2]))
		self.assertTrue(len(pre_class_indices[target_cluster] + 1), len(post_class_indices[target_cluster]))
		self.assertTrue(np.array_equal(np.append(pre_class_indices[target_cluster], 43), post_class_indices[target_cluster]))

		self.assertEqual(pre_param_vals, post_param_vals)

	def test_nonsingleton_different_new_class(self):
		x, y, _ = BaseTest.get_data()
		imgpe = iMGPE(x, y, initial_k=6)
		BaseTest.divide_data(imgpe, singleton=False)
		BaseTest.prepare_kernels(imgpe)
		target_cluster = 6 # Doesn't exist yet
		BaseTest.boost_target_affinity(imgpe, 43, target_cluster)
		BaseTest.boost_priors(imgpe)

		pre_c, pre_class_indices, pre_param_vals = BaseTest.extract_sample_state(imgpe)
		imgpe.gibbs(43)
		post_c, post_class_indices, post_param_vals = BaseTest.extract_sample_state(imgpe)

		self.assertEqual(len(pre_c), len(post_c))
		for index, (pre_c_i, post_c_i) in enumerate(zip(pre_c, post_c)):
			if index == 43:
				self.assertEqual(post_c_i, target_cluster)
			else:
				self.assertEqual(pre_c_i, post_c_i)

		pre_class_indices_compare = [pre_class_indices_j for index, pre_class_indices_j in enumerate(pre_class_indices) if index != 2]
		post_class_indices_compare = [pre_class_indices_j for index, pre_class_indices_j in enumerate(post_class_indices) if index != 2 and index != target_cluster]
		self.assertEqual(len(pre_class_indices) + 1, len(post_class_indices))
		for pre_class_indices_j, post_class_indices_j in zip(pre_class_indices_compare, post_class_indices_compare):
			self.assertTrue(np.array_equal(pre_class_indices_j, post_class_indices_j))
		self.assertTrue(len(pre_class_indices[2]) - 1, len(post_class_indices[2]))
		self.assertTrue(np.array_equal(pre_class_indices[2][pre_class_indices[2] != 43], post_class_indices[2]))
		self.assertTrue(1, len(post_class_indices[target_cluster]))
		self.assertTrue(np.array_equal(np.array([43]), post_class_indices[target_cluster]))

		self.assertEqual(len(pre_param_vals) + 1, len(post_param_vals))
		post_param_vals_compare = [post_param_vals_j for index, post_param_vals_j in enumerate(post_param_vals) if index != 6]
		self.assertEqual(pre_param_vals, post_param_vals_compare)

if __name__ == '__main__':
	unittest.main()