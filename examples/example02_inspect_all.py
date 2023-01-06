"""Make plots for example 2.

Author:
    Ilias Bilionis

Date:
    1/2/2023

"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
sns.set_style("ticks")
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 9,
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
    prefix = f"example02_newa_results/example02_gamma={gamma:1.2e}_n={n}_sigma={sigma:1.2e}_0"
    thetas.append(np.loadtxt(prefix + "_theta.csv")[:, 1])
thetas = np.vstack(thetas)

skip = 5000
thin = 100

thetas = thetas[:, skip::thin]

betas = np.exp(thetas)
p_500, p025, p975 = np.percentile(thetas, [50, 2.5, 97.5], axis=1)

fig, ax = plt.subplots(figsize=(3.543, 3.543/1.618))
#sns.boxplot(thetas.T, width=0.1, fliersize=0, linewidth=1, whis=[5, 95])

ax.boxplot(thetas[:, :].T, showbox=False, whis=[5, 95], showfliers=False,
           widths=0.05, capwidths=0)
#ax.set_yscale('log')

ax.set_ylabel("$\\beta$ posterior\nquantiles", rotation="horizontal")
#print(ax.yaxis.label_coords)
ax.yaxis.set_label_coords(0.12, 0.9)
labels = [f"{g:1.2f}" for g in gammas]
labels[0] = labels[0] + "\nwrong model"
labels[-1] = labels[-1] + "\ncorrect model"
ax.set_xticklabels(labels)
ax.set_xlabel("$\gamma$ (model correctness)")
yticks = np.linspace(p025.min(), p975.max(), 5)
ax.set_yticks(yticks)
ylabels = []
for b in np.exp(yticks):
    tmp = f"{b:1.1e}".split("e")
    l = f"${tmp[0]}\\times 10^{tmp[1][2:]}$"
    ylabels.append(l)
ax.set_yticklabels(ylabels)

sns.despine(trim=True)
#plt.legend(loc="best", frameon=False)

plt.tight_layout()
plt.savefig("example02b_prelim.png")
plt.savefig("example02b_prelim.eps")
plt.savefig("example02b_prelim.pdf")
plt.show()
    

