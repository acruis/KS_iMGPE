from ks_imgpe import KSiMGPE
import numpy as np
import math
import GPy
from GPy.kern import Add
from prod import Prod

class KSiMGPEABCD(KSiMGPE):

	# ------
	# ABCD
	# ------

	def create_base_kernel_set(self):
		return [
			self.create_rbf_kernel(),
			self.create_white_kernel(),
			self.create_periodic_kernel(),
			self.create_linear_kernel(),
			self.create_constant_kernel()
		]

	def expand_root(self, kernel_tree, kernels):
		results = []
		if not (isinstance(kernel_tree, Prod) or isinstance(kernel_tree, Add)):
			# Base kernel
			results.extend([kernel.copy() for kernel in kernels if not (type(kernel) is type(kernel_tree))])
		results.extend([Add([kernel_tree.copy(), kernel.copy()]) for kernel in kernels])
		results.extend([Prod([kernel_tree.copy(), kernel.copy()]) for kernel in kernels])
		return results

	def expand_tree(self, kernel_tree, kernels):
		results = self.expand_root(kernel_tree, kernels)
		
		# Base kernel
		if not (isinstance(kernel_tree, Prod) or isinstance(kernel_tree, Add)):
			return results
		
		# Is composite kernel
		component_kernels = kernel_tree.parameters
		for i, component_kernel in enumerate(component_kernels):
			expanded_component_results = self.expand_tree(component_kernel, kernels)
			for expanded_component_result in expanded_component_results:
				component_kernels_with_expansion = component_kernels[:i] + [expanded_component_result] + component_kernels[i+1:]
				component_kernels_with_expansion = [kern.copy() for kern in component_kernels_with_expansion]
				results.append(kernel_tree.__class__(component_kernels_with_expansion))
		
		return results

	def replace_kernel(self, j, kernel):
		base_kernels = self.create_base_kernel_set()
		replacement_kernels = self.expand_tree(kernel, base_kernels)
		replacement_kernels.append(kernel)
		for replacement_kernel in replacement_kernels:
			self.mle(j, kernel=replacement_kernel)
		replacement_kernel_scores = [self.measure_bic(j, replacement_kernel) for replacement_kernel in replacement_kernels]
		return sorted(zip(replacement_kernels, replacement_kernel_scores), key=lambda (kern, bic):bic)[0][0]

	def replace_kernels(self):
		for j, existing_kernel in enumerate(self.kernels):
			self.kernels[j] = self.replace_kernel(j, existing_kernel)

	# -----------------------------------------
	# Chain iteration with kernel replacement
	# -----------------------------------------

	def iterate_chain(self, iterations=1, skips=-1, save_output=False, output_folder='.'):
		for it in range(iterations):
			print it
			self.gibbs_round()
			self.hmc_round()
			self.optimize_priors()
			self.ars_sample_alpha()
			self.hmh_gating_phi()

			if (it % 300) == 0 and (it > 0):
				self.replace_kernels()

			if save_output and (skips > 0) and (it % skips == 0):
				self.save_model(filename=("%d_ks_abcd.txt" % it), folder=output_folder)
