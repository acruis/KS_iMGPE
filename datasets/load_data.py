import numpy as np

def load_motorcycle():
	times = []
	accels = []

	with open('datasets/motorcycle') as f:
		for line in f:
			split_line = line.split()
			times.append(float(split_line[0]))
			accels.append(float(split_line[1]))

	return np.array(times).reshape(len(times),1), np.array(accels).reshape(len(times),1)

def load_toy():
	import json
	with open('datasets/toy_data') as f:
		xy = json.load(f)
		return np.array(xy['x']).reshape(len(xy['x']), 1), np.array(xy['y']).reshape(len(xy['y']), 1)

def strip_test_data(xy):
	x, y = xy
	new_x = []
	new_y = []
	for i in range(len(x)):
		if i % 5 == 2:
			continue
		new_x.append(x[i])
		new_y.append(y[i])
	return np.array(new_x), np.array(new_y)