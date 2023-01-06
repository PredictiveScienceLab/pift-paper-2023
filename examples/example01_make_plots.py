"""Make plots for example 1.

Author:
    Ilias Bilionis

Date:
    1/3/2023

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

# True solution
x = np.linspace(0, 1, 100)
k = 0.25
ya = 1.0
yb = 0.1
c1 = ya + 1./k
c2 = yb - ya + (np.exp(-1.) - 1.) / k
f = lambda x: c1 + c2 * x - 1 / k * np.exp(-x)

# Make prediction plots
print("*** Making plots")
betas = [100, 1000, 10000, 100000]
for b in betas:
    prefix = f"example01_results/example01_beta={b:1.2e}"
    xs = np.loadtxt(prefix + "_x.csv")
    ys = np.loadtxt(prefix + "_phi.csv")
    p500, p025, p975 = np.percentile(ys, [50, 2.5, 97.5], axis=0)
    fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
    ax.plot(xs, p500, lw=0.5, label="Posterior predictive interval")
    ax.fill_between(xs, p025, p975, alpha=0.5, color=sns.color_palette()[0])
    lines = ax.plot(xs, ys[::1000, :].T, color=sns.color_palette()[1], lw=0.3)
    lines[0].set_label("Posterior samples")
    ax.plot(xs, f(xs), '--', lw=0.5, color=sns.color_palette()[2], label="Ground truth")
    yticks = np.linspace(p025.min(), p975.max(), 5)
    ax.set_yticks(yticks)
    ylabels = []
    for y in yticks:
        ylabels.append(f"{y:1.1f}")
    ax.set_yticklabels(ylabels)
    ax.set_ylim(0.1, 1.4)
    ax.set_xlabel("$x$")
    tmp = f"{b:1.1e}".split("e")
    l = f"10^{tmp[1][2:]}"
    ax.set_title(f"$\\beta={l}$")
    if b == 100 or b == 10000:
        ax.set_ylabel("$y$")
    if b == 100000:
        plt.legend(loc="best", frameon=False)
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    file = prefix + ".pdf"
    print("> writing: " + file)
    plt.savefig(prefix + ".pdf")
    plt.savefig(prefix + ".png", dpi=150)
