"""Compare true solutions for various choices of kappa.

Author:
    Ilias Bilionis

Date:
    1/3/2023
"""


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
sns.set_style("white")

import numpy as np

x1 = np.loadtxt("example02b_kappa=0.00e+00_xs.csv")
y1 = np.loadtxt("example02b_kappa=0.00e+00_ys.csv")
x2 = np.loadtxt("example02b_kappa=1.00e+00_xs.csv")
y2 = np.loadtxt("example02b_kappa=1.00e+00_ys.csv")


fig, ax = plt.subplots()
ax.plot(x1, y1, label="$\kappa=0$")
ax.plot(x2, y2, label="$\kappa=1$")
plt.legend(loc="best", frameon=False)
sns.despine(offset=10, trim=True)
plt.savefig("example02b_sanity_check.pdf")
plt.show()
