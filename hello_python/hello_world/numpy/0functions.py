import numpy as np

def square(x):
	return np.power(x, 2)
def relu(x):
	return np.maximum(0, x)
def leaky_relu(x):
	return np.maximum(0.2*x, x)

a = np.array([[1, 1, 1],[2, 2, 2],[3, 3, 3]])
print(square(a))