"""Make the figures of example 3.

Author:
    Ilias Bilionis

Date:
    1/3/2023

"""

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 9,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params) 
import numpy as np
import sys

# Extract parameters from command line
beta = float(sys.argv[1])
n = int(sys.argv[2])
sigma = float(sys.argv[3])
folder = sys.argv[4]

prefix = folder + f"/example03_beta={beta:1.2e}_n={n}_sigma={sigma:1.2e}_0"

# Warmup period
#skip = 2000
skip = 5000
# How many samples to skip
thin = 100

# Load the parameters
thetas = np.loadtxt(prefix + "_theta.csv")

# Plot the evolution of the Markov chains (for sanity check)
fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
ax.plot(thetas[:, 1:], lw=0.5)
theta_min = thetas[:, 1:].min()
theta_max = thetas[:, 1:].max()
ax.plot([skip, skip], [theta_min, theta_max],
        "k--", lw=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Parameters", rotation="horizontal")
ax.yaxis.set_label_coords(0., 0.9)
ax.text(skip + 100, theta_min, "End of warmup period")
ax.text(9000, -0.3, "$\log(\kappa)$", color=sns.color_palette()[1])
ax.text(9000, -2.0, "$\log(D)$", color=sns.color_palette()[0])
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
print("> writing: " + prefix + "_theta.pdf")
plt.savefig(prefix + "_theta.pdf")
plt.savefig(prefix + "_theta.png", dpi=150)

# Plot posterior over D and kappa
fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
D = np.exp(thetas[skip::thin, 1])
kappa = np.exp(thetas[skip::thin, 2])
sns.kdeplot(x=D, y=kappa, linewidths=0.5, 
            colors=[sns.color_palette()[1]])
ax.plot([0.1], [1.0], 'x', color=sns.color_palette()[2], label="Ground truth")
ax.set_xlabel("$D$")
ax.set_ylabel("$\kappa$", rotation="horizontal")
ax.set_xlim(0., 0.3)
ax.set_ylim(0.5, 1.5)
ax.text(0.105, 1.2, "Posterior", color=sns.color_palette()[1]) 
ax.text(0.105, 1.01, "Ground truth", color=sns.color_palette()[2])
#plt.legend(loc="best", frameon=False)
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
print("> writing: " + prefix + "_theta_post.pdf")
plt.savefig(prefix + "_theta_post.pdf")
plt.savefig(prefix + "_theta_post.png", dpi=150)

# Plot posterior of source
try:
    x = np.loadtxt(prefix + "_source_x.csv")
    y = np.loadtxt(prefix + "_source_f.csv")
    fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
    med, low, up = np.percentile(y, [50, 2.5, 97.5], axis=0)
    #ax.plot(x, med, color=sns.color_palette()[0], lw=0.5)
    #ax.fill_between(x, low, up, color=sns.color_palette()[0], alpha=0.25)
    l = ax.plot(x, y[skip::thin,:].T, color=sns.color_palette()[1], lw=0.3)
    ax.plot(x, np.cos(4.0 * x), '--', lw=0.5, color=sns.color_palette()[2])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$", rotation="horizontal")
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    plt.savefig(prefix + "_source.pdf")
except:
    print("No source term found.")

# Plot the prior after calibrating the parameters
obs_prefix = f"example02_n={n}_sigma={sigma:1.2e}_0"
x_file = obs_prefix + "_x_obs.csv"
y_file = obs_prefix + "_y_obs.csv"
xs = np.loadtxt(prefix + "_prior_x.csv")
phipr = np.loadtxt(prefix + "_prior_phi.csv")
phips = np.loadtxt(prefix + "_post_phi.csv")

x_obs = np.loadtxt(x_file)
y_obs = np.loadtxt(y_file)

xst = np.loadtxt("example02_xs.csv")
yst = np.loadtxt("example02_ys.csv")

fig, ax = plt.subplots(figsize=(7.48/2, 7.48/2 / 1.618))
med, low, up = np.percentile(phipr, [50, 2.5, 97.5], axis=0)
#ax.plot(xs, med, color=sns.color_palette()[0], lw=0.5)
#ax.fill_between(xs, low, up, color=sns.color_palette()[0], alpha=0.25)
ax.plot(xs, phipr[skip::thin*10,:].T, lw=0.3, color=sns.color_palette()[1])
ax.plot(x_obs, y_obs, 'ko', markeredgewidth=0.5, markersize=1)
ax.plot(xst, yst, '--', lw=1.0, color=sns.color_palette()[2])
ax.set_title("Fitted prior predictive")
ax.set_xlabel("$x$")
ax.set_ylabel("$\phi(x)$", rotation="horizontal")
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
file_pre = prefix + "_fitted_prior_predictive"
print("> writing: " + file_pre + ".pdf")
plt.savefig(file_pre + ".pdf")
plt.savefig(file_pre + ".png", dpi=150)

fig, ax = plt.subplots(figsize=(7.48/2, 7.48/2 / 1.618))
med, low, up = np.percentile(phips, [50, 2.5, 97.5], axis=0)
#ax.plot(xs, med, color=sns.color_palette()[0], lw=0.5, label="Predictive interval")
#ax.fill_between(xs, low, up, color=sns.color_palette()[0], alpha=0.25)
lines = ax.plot(xs, phips[skip::thin*10,:].T, lw=0.3, color=sns.color_palette()[1])
lines[0].set_label("Predictive samples")
ax.plot(xst, yst, '--', lw=1, color=sns.color_palette()[2], label="Ground truth")
ax.plot(x_obs, y_obs, 'k.', markeredgewidth=0.5, markersize=1, label="Observations")
ax.set_title("Posterior predictive")
ax.set_xlabel("$x$")
#ax.set_ylabel("$\phi(x)$", rotation="horizontal")
plt.legend(loc="best", frameon=False)
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
file_pre = prefix + "_fitted_post_predictive"
print("> writing: " + file_pre + ".pdf")
plt.savefig(file_pre + ".pdf")
plt.savefig(file_pre + ".png", dpi=150)


plt.show()
