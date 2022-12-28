"""Make some plots for test_prior_exp.cpp.

Author:
    Ilias Bilionis

Date:
    12/28/2022

"""

import matplotlib.pyplot as plt
import numpy as np

x = np.loadtxt("example02_gamma=1.00e+00_x.csv")
y = np.loadtxt("example02_gamma=1.00e+00_phi.csv")

fig, ax = plt.subplots()
ax.plot(x, y[-100:,:].T, 'r', lw=0.05)
plt.show()
