"""Make some plots for test_post_exp.cpp.

Author:
    Ilias Bilionis

Date:
    12/28/2022

"""

import matplotlib.pyplot as plt
import numpy as np

x_obs = np.loadtxt("../examples/example02_n=10_sigma=1.00e-02_0_x_obs.csv")
y_obs = np.loadtxt("../examples/example02_n=10_sigma=1.00e-02_0_y_obs.csv")
x = np.loadtxt("example02_gamma=1.00e+00_x.csv")
y = np.loadtxt("example02_gamma=1.00e+00_phi.csv")

fig, ax = plt.subplots()
ax.plot(x, y[-100:,:].T, 'r', lw=0.05)
ax.plot(x_obs, y_obs, 'xk')
plt.show()
