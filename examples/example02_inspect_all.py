"""Make plots for example 2.

Author:
    Ilias Bilionis

Date:
    1/2/2023

"""

import matplotlib.pyplot as plt
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params) 

import numpy as np
import math
import sys

n = int(sys.argv[1])
sigma = float(sys.argv[2])

thetas = []
gammas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
for gamma in gammas:
    prefix = f"example02_resultsa/example02_gamma={gamma:1.2e}_n={n}_sigma={sigma:1.2e}_0"
    thetas.append(np.loadtxt(prefix + "_theta.csv")[:, 1])
thetas = np.array(thetas)

skip = 200

thetas = thetas[:, skip:]

betas = np.exp(thetas)
p_500, p025, p975 = np.percentile(thetas, [50, 2.5, 97.5], axis=1)

fig, ax = plt.subplots()
ax.boxplot(thetas[:, :].T, showbox=False, whis=[5, 95], showfliers=False,
           widths=0.05, capwidths=0)
#ax.set_yscale('log')

#ax.set_xticks(gammas)
ax.set_ylabel("$\\beta$ posterior\nquantiles", rotation="horizontal")
labels = [f"{g:1.2f}" for g in gammas]
labels[0] = labels[0] + "\nwrong model"
labels[-1] = labels[-1] + "\ncorrect model"
ax.set_xticklabels(labels)
ax.set_xlabel("$\gamma$ (model correctness)")
ax.set_yticks(p_500[1:])
ylabels = []
for b in np.exp(p_500[1:]):
    tmp = f"{b:1.1e}".split("e")
    l = f"${tmp[0]}\\times 10^{tmp[1][2:]}$"
    ylabels.append(l)
ax.set_yticklabels(ylabels)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
#ax.spines['bottom'].set_bounds(0, 1)
ax.spines['left'].set_visible(True)
#ax.spines['left'].set_bounds(, ymax)
plt.tight_layout()
plt.savefig("example02_prelim.png")
plt.savefig("example02_prelim.eps")
plt.savefig("example02_prelim.pdf")
plt.show()
    

