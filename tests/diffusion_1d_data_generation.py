"""Test data generation in the 1D diffusione example."""


import numpy as np
import matplotlib.pyplot as plt

from problems import Diffusion1D

example = Diffusion1D(
    kappa = 0.1,
    yb = (1.0, 0.0)
)

result = example.generate_experimental_data(sigma=0.1)
x = result["x"]
y = result["y"]
f = result["solution"]

fig, ax = plt.subplots()
xs = np.linspace(example.a, example.b, 100)
ys = f(xs)
ax.plot(xs, ys, 'b--', lw=2)
ax.plot(x, y, 'kx')
plt.show()
