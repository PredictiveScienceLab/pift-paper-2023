"""Test some of the utilities."""

import matplotlib.pyplot as plt
import numpy as np

from pift import uniformly_sample_1d_function_w_Gaussian_noise

f = lambda x: x ** 3 + 0.5 * x ** 2

fig, ax = plt.subplots()
xs = np.linspace(-1.0, 1.0, 100)
ys = f(xs)

ax.plot(xs, ys)

np.random.seed(123456)

xr, yr = uniformly_sample_1d_function_w_Gaussian_noise(
    f,
    10,
    a=-1.0,
    b=1.0,
    sigma=0.1
)

ax.plot(xr, yr, 'kx')

plt.show()
