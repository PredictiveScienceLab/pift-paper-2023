"""Test the Basis classes."""

import matplotlib.pyplot as plt
import numpy as np

from pift import Fourier1DBasis, Fourier2DBasis, eval_on_grid

# Test 1D
basis = Fourier1DBasis(1.0, 3)

xs = np.linspace(0, basis.L, 100)
ys = basis(xs)

fig, ax = plt.subplots()
ax.plot(xs, ys)
#plt.show()

# Test 2D
basis2d = Fourier2DBasis(1.0, 2)
X, Y = np.meshgrid(*(np.linspace(0, l, 100) for l in basis2d.L))
Z = eval_on_grid(basis2d, [X, Y])

levels = np.linspace(Z.min(), Z.max(), 10)
for i in range(basis2d.dim):
    fig, ax = plt.subplots()
    c = ax.contourf(X, Y, Z[:, :, i], levels=levels)
    plt.colorbar(c)
plt.show()
