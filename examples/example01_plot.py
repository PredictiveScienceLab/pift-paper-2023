"""This plots the results for example 1.

Author:
    Ilias Bilonis

Date:
    12/21/2022

"""

import matplotlib.pyplot as plt
import sys
import numpy as np
import yaml


if len(sys.argv) != 3:
    print(f"Usage:\n\t python {sys.argv[0]} <BETA> <CONFIG_FILE>")
    exit(1)

beta = float(sys.argv[1])
assert beta > 0.
config_file = sys.argv[2]
with open(config_file) as fd:
    config = yaml.load(fd.read(), yaml.Loader)

prefix = config["output"]["prefix"] + f"_beta={beta:.2e}"
warmup_file = prefix + "_warmup.csv"
samples_file = prefix + "_samples.csv"
x_file = prefix + "_x.csv"
phi_file = prefix + "_phi.csv"

data_warmup = np.loadtxt(warmup_file)
hs_warmup = data_warmup[:, 0]
ws_warmup = data_warmup[:, 1:]
data_samples = np.loadtxt(samples_file)
hs_samples = data_samples[:, 0]
ws_samples = data_samples[:, 1:]
num_samples = ws_samples.shape[0]
xs = np.loadtxt(x_file)
n = xs.shape[0]
phis = np.loadtxt(phi_file)
assert num_samples == phis.shape[0]
assert n == phis.shape[1]

# TODO: Alex are the right options for plotting nicely and remove this comment

# The evolution of the samples from the Hamiltonian during training
hs = np.hstack([hs_warmup, hs_samples])
fig, ax = plt.subplots()
ax.plot(hs)
hs_file = prefix + "_hs.png"
print("\t- ", hs_file)
plt.savefig(hs_file)

# All the samples that we took
ws = np.vstack([ws_warmup, ws_samples])
fig, ax = plt.subplots()
ax.plot(ws)
ws_file = prefix + "_ws.png"
print("\t- ", ws_file)
plt.savefig(ws_file)

# The field samples
phi_500, phi_025, phi_975 = np.percentile(phis, [50, 2.5, 97.5], axis=0)
fig, ax = plt.subplots()
ax.plot(xs, phi_500, 'b', label="Median")
ax.fill_between(xs, phi_025, phi_975, color="blue", alpha=0.25)
ax.plot(xs, phis[::100, :].T, 'r', lw=0.1)
phis_file = prefix + "_phis.png"
print("\t- ", phis_file)
plt.savefig(phis_file)

plt.show()

