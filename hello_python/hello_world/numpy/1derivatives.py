import numpy as np
from typing import Callable

def square(x):
	return np.power(x, 2)
def relu(x):
	return np.maximum(0, x)
def leaky_relu(x):
	return np.maximum(0.2*x, x)

a = np.array([[1, 1, 1],[2, 2, 2],[3, 3, 3]])
print(square(a))

def deriv(func, input_, delta):
	return (func(input_+delta)-func(input_-delta))/(2.0*delta)

print(deriv(relu, 3, 0.0001))