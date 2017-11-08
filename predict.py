from models.imgpe import iMGPE
from models.ks_imgpe_sample import KSiMGPESample
from models.ks_imgpe_mle import KSiMGPEMLE
from models.ks_imgpe_abcd import KSiMGPEABCD

from datasets.load_data import load_motorcycle, strip_test_data

x, y = strip_test_data(load_motorcycle())

# Any of the below will work.
# m = iMGPE(x, y)
# m = KSiMGPESample(x, y)
# m = KSiMGPEMLE(x, y)
m = KSiMGPEABCD(x, y)

import matplotlib.pyplot as plt
import numpy as np

x_star = np.linspace(0, 60, 80)

for _ in range(5):
	m.iterate_chain(iterations=1)
	y_star = m.predict(x_star)
	plt.plot(x_star, y_star)

plt.show()