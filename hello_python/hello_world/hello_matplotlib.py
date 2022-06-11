import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


x = np.array([-2, -1, 0, 1, 2])

figure = plt.figure()

ax1 = figure.add_subplot(1, 1, 1)

ax1.plot(x, np.power(x, 2), color="blue", linestyle="-", marker="*", label="y=x^2")
ax1.set_title("Title", fontsize=20)
ax1.set_xlabel("Ось х")
ax1.set_ylabel("Ось y")
ax1.legend(loc="best")

figure.savefig("./fig.png")

plt.show()