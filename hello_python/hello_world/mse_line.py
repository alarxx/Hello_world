import numpy as np
import matplotlib.pyplot as plt


y = np.array([1, 1, 1, 2, 2, 2])
x = range(len(y))

z = np.polyfit(x, y, 1)
p = np.poly1d(z)

for i in x:
    print(p(i))