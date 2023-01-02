import matplotlib.pyplot as plt
import numpy as np


sigma = 0.0001
t1 = np.loadtxt("example02b_gamma=1.00e+00_theta.csv")
beta1 = np.exp(t1[100:, 1]) * sigma ** 2
t2 = np.loadtxt("example02b_gamma=0.00e+00_theta.csv")
beta2 = np.exp(t2[100:, 1]) * sigma ** 2
t3 = np.loadtxt("example02b_gamma=5.00e-01_theta.csv")
beta3 = np.exp(t3[100:, 1]) * sigma ** 2

fig, ax = plt.subplots()
ax.plot(t1[:, 1], alpha=0.25)
ax.plot(t2[:, 1], alpha=0.25)
ax.plot(t3[:, 1], alpha=0.25)


fig, ax = plt.subplots()
ax.plot(beta1, lw=0.5)
ax.plot(beta2, lw=0.5)
ax.plot(beta3, lw=0.5)

#fig, ax = plt.subplots()
#ax.hist(beta1, alpha=0.25)
#ax.hist(beta2, alpha=0.25)
#ax.hist(beta3, alpha=0.25)
plt.show()
