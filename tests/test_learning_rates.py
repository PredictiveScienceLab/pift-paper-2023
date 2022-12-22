"""Plotting different learning rates for SGLD to build intuition.

Author:
    Ilias Bilionis

Date:
    12/22/2022

"""


import matplotlib.pyplot as plt
import numpy as np


t = np.arange(0, 1_000_000)


alpha = 1e-4
beta = 1.0
gamma = 0.51
delta = 10_000

e1 = alpha / (beta + t) ** gamma
e2 = alpha / (beta + t / delta) ** gamma
e3 = alpha * delta ** gamma / (beta * delta + t) ** gamma

print(f"Original learning rate {alpha:1.2e}")
print(f"Effective learning rate {alpha * delta ** gamma:1.2e}")

fig, ax = plt.subplots()
ax.plot(e1, label="Learning rate 1")
ax.plot(e2, '--', label="Learning rate 2")
ax.plot(e3, '-.', label="Learning rate 3")
ax.set_yscale("log")
plt.legend(loc="best")
plt.show()
