from ks_imgpe import KSiMGPE
import numpy as np
import math
import GPy
from GPy.kern import Add
from prod import Prod

class KSiMGPEMLE(KSiMGPE):

	# ----------------------------
	# MLE step (in place of HMC)
	# ----------------------------

	def mle_round(self):
		for j in range(len(self.class_indices)):
			self.mle(j)

	# --------------------
	# Kernel replacement
	# --------------------

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
		bic_existing = self.measure_bic(j, existing_kernel)
		bic_replacement = self.measure_bic(j, replacement_kernel)
		return bic_replacement < bic_existing # Smaller BIC is better

	def replace_kernels(self):
		for j, existing_kernel in enumerate(self.kernels):
			replacement_kernel = self.generate_replacement_kernel(existing_kernel)
			self.mle(j, kernel=replacement_kernel)
			if self.compare_replacement_kernel(j, existing_kernel, replacement_kernel):
				self.kernels[j] = replacement_kernel

	# -----------------------------------------
	# Chain iteration with kernel replacement
	# -----------------------------------------

	def iterate_chain(self, iterations=1, skips=-1, save_output=False, output_folder='.'):
		for it in range(iterations):
			print it
			self.gibbs_round()
			self.mle_round()
			self.replace_kernels()
			self.optimize_priors()
			self.ars_sample_alpha()
			self.hmh_gating_phi()

			if save_output and (skips > 0) and (it % skips == 0):
				self.save_model(filename=("%d_ks_mle.txt" % it), folder=output_folder)
