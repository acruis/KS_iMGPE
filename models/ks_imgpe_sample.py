from ks_imgpe import KSiMGPE
import numpy as np
import math
import GPy
from GPy.kern import Add
from prod import Prod

class KSiMGPESample(KSiMGPE):
	
	# --------------------
	# Kernel replacement
	# --------------------

	def pick_kernel(self):
		k = len(self.class_indices)
		j = int(np.random.random() * k)
		return j, self.kernels[j]

	def generate_replacement_kernel(self, existing_kernel):
		operators = ['add', 'mult', 'replace']
		picked_operator = np.random.choice(operators)
		base_kernel = self.create_base_kernel()

		if picked_operator == 'add':
			return Add([existing_kernel.copy(), base_kernel.copy()])
		elif picked_operator == 'mult':
			return Prod([existing_kernel.copy(), base_kernel.copy()])

		assert picked_operator == 'replace'
		return base_kernel

	def compare_replacement_kernel(self, j, existing_kernel, replacement_kernel):
		# MLE first before measuring BIC
		existing_param_array = existing_kernel.param_array.astype(np.float64)
		replacement_param_array = replacement_kernel.param_array.astype(np.float64)
		self.mle(j, kernel=existing_kernel)
		self.mle(j, kernel=replacement_kernel)

		bic_existing = self.measure_bic(j, existing_kernel)
		bic_replacement = self.measure_bic(j, replacement_kernel)

		# Return kernel params to their pre-MLE values.
		existing_kernel[:] = existing_param_array
		replacement_kernel[:] = replacement_param_array
		return bic_replacement < bic_existing # Smaller BIC is better

	def replace_kernels(self):
		for j, existing_kernel in enumerate(self.kernels):
			replacement_kernel = self.generate_replacement_kernel(existing_kernel)
			if self.compare_replacement_kernel(j, existing_kernel, replacement_kernel):
				self.kernels[j] = replacement_kernel

	# -----------------------------------------
	# Chain iteration with kernel replacement
	# -----------------------------------------

	def iterate_chain(self, iterations=1, skips=-1, save_output=False, output_folder='.'):
		for it in range(iterations):
			print it
			self.gibbs_round()
			self.hmc_round()
			self.replace_kernels()
			self.optimize_priors()
			self.ars_sample_alpha()
			self.hmh_gating_phi()

			if save_output and (skips > 0) and (it % skips == 0):
				self.save_model(filename=("%d_ks_sample.txt" % it), folder=output_folder)
