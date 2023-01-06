"""Make the figures of example 3.

Author:
    Ilias Bilionis

Date:
    1/3/2023

"""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks")
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 9,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params) 
from statsmodels.graphics import tsaplots
import numpy as np
import sys

beta = float(sys.argv[1])
n = int(sys.argv[2])
sigma = float(sys.argv[3])

prefix = f"example03b_beta={beta:1.2e}_n={n}_sigma={sigma:1.2e}_0"

skip = 500
thin = 30

thetas = np.loadtxt(prefix + "_theta.csv")

# Plot posterior of parameters
fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
ax.plot(thetas[:, 1:], lw=0.5)
theta_min = thetas[:, 1:].min()
theta_max = thetas[:, 1:].max()
ax.plot([skip, skip], [theta_min, theta_max],
        "k--", lw=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("Parameters")
ax.text(skip + 100, theta_min, "End of warmup period")
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
plt.savefig(prefix + "_theta.pdf")


# fig, ax = plt.subplots()
# ax.plot(np.exp(thetas[:, 1]), np.exp(thetas[:,2]), lw=0.3)
# ax.plot(0.1, 1.0, 'kx')

#plt.show()
#quit()

# Autocorrelation plot
for i in range(1, thetas.shape[1]):
    fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
    tsaplots.plot_acf(thetas[skip:, 1], ax=ax, lags=np.arange(0, 200, 10),
                      alpha=None,
                      title=f"Autocorrelation $\\theta_{{{i+1}}}$")
    sns.despine(trim=True)
    plt.tight_layout(pad=0.1)
    
thetas = thetas[skip::thin, :]


fig, ax = plt.subplots(figsize=(7.48 / 2, 7.48/2/1.618))
D = np.exp(thetas[:, 1])
kappa = np.exp(thetas[:, 2])
sns.kdeplot(x=D, y=kappa, label="Posterior", linewidths=0.5, 
            colors=[sns.color_palette()[1]])
#ax.scatter(D, kappa, s=2, marker='.', color=sns.color_palette()[1])
ax.plot([0.1], [1.0], 'x', color=sns.color_palette()[2], label="Ground truth")
ax.set_xlabel("$D$")
ax.set_ylabel("$\kappa$")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
plt.savefig(prefix + "_theta_post.pdf")

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
ax.plot(xst, yst, '--', lw=0.5, color=sns.color_palette()[2])
med, low, up = np.percentile(phipr, [50, 2.5, 97.5], axis=0)
ax.plot(xs, med, color=sns.color_palette()[0], lw=0.5)
ax.fill_between(xs, low, up, color=sns.color_palette()[0], alpha=0.25)
ax.plot(xs, phipr[skip::thin*5,:].T, lw=0.3, color=sns.color_palette()[1])
ax.plot(x_obs, y_obs, 'kx', markeredgewidth=0.5, markersize=1)
ax.set_title("Fitted prior predictive")
ax.set_xlabel("$x$")
ax.set_ylabel("$\phi(x)$")
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
plt.savefig(prefix + "_fitted_prior_predictive.pdf")

fig, ax = plt.subplots(figsize=(7.48/2, 7.48/2 / 1.618))
ax.plot(xst, yst, '--', lw=1)
med, low, up = np.percentile(phips, [50, 2.5, 97.5], axis=0)
ax.plot(xs, med, color=sns.color_palette()[0], lw=0.5)
ax.fill_between(xs, low, up, color=sns.color_palette()[0], alpha=0.25)
ax.plot(xs, phips[skip::thin*5,:].T, lw=0.3, color=sns.color_palette()[1])
ax.plot(x_obs, y_obs, 'k.', markeredgewidth=0.5, markersize=1)
ax.set_title("Posterior predictive")
ax.set_xlabel("$x$")
ax.set_ylabel("$\phi(x)$")
sns.despine(trim=True)
plt.tight_layout(pad=0.1)
plt.savefig(prefix + "_posterior_predictive.pdf")

plt.show()
