"""Make plots for example 1.

Author:
    Ilias Bilionis

Date:
    1/3/2023

"""

import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_context("paper")
#sns.set_style("white")
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
    plt.tight_layout(pad=0.1)
    sns.despine(trim=True)
    plt.savefig(prefix + ".pdf")


# Make figure that depicts uncertainty
data = []
for b in [1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    prefix = f"example01_results/example01_beta={b:1.2e}"
    xs = np.loadtxt(prefix + "_x.csv")
    i = xs.shape[0] // 2
    x = xs[i]
    ys = np.loadtxt(prefix + "_phi.csv")
    y = ys[:, i]
    data.append(y)
data = np.array(data)
fig, ax = plt.subplots(figsize=(7.48, 7.48/1.618))
ax.boxplot(data.T, showbox=False, whis=[5, 95], showfliers=False,
           widths=0.05, capwidths=0)
ax.plot(np.arange(1, 7), f(x) * np.ones((6,)), '--', lw=0.5, color=sns.color_palette()[2])
ax.set_ylabel("$\phi(0.5)$", rotation="horizontal")
ax.set_xlabel("$\\beta$")
l, u = np.percentile(data, [5., 95.], axis=1)
ax.set_yticks([l.min(), f(x), u.max()])
ax.set_xticklabels(["$1$", "$10^2$", "$10^3$", "$10^4$", "$10^5$", "$10^6$"])
plt.tight_layout(pad=0.1)
sns.despine(trim=True)
plt.show()
