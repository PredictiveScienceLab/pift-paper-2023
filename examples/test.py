import numpy as np
import matplotlib.pyplot as plt

wpr = np.loadtxt("example02b_gamma=1.00e+00_post_ws.csv")
wps = np.loadtxt("example02b_gamma=1.00e+00_prior_ws.csv")

for i in range(1, wpr.shape[1]):
    fig, ax = plt.subplots()
    ax.plot(wpr[:, i], lw=1)
    ax.plot(wps[:, i], lw=1)

fig, ax = plt.subplots()
ax.plot(wpr[:, 1:], lw=1)
ax.plot(wps[:, 1:], lw=1)
plt.show()
