# Reproducing the results of Example 2

## Objective
The objective of this example is to demonstrate how physics-informed information
field theory can be used to solve inverse problems.

## Mathematical details
In this example the spatial domain is $[0, 1]$.
The ground truth field satisfies the equation:

$$
D\frac{d^2\phi}{dx^2} - \kappa \phi^3  = f,
$$

with source term:

$$
f(x) = \cos(4x),
$$

conductivity:

$$
D = 0.1,
$$

non-linear coefficient:

$$
\kappa = 1,
$$

and boundary conditions:

$$
\phi(0) = 0, 
$$

and

$$
\phi(1) = 0.
$$

We sample the field at $40$ equidistant points between $0$ and $1$ and we add
Gaussian noise with zero mean and standard deviation $0.01$.
The observed field inputs are [here](example02_n=40_sigma=1.00e-02_0_x_obs.csv)
and the observed field otuputs are [here](example02_n=40_sigma=1.00e-02_0_x_obs.csv).
If you wish to review how the observations were generated, consult the script
[example02_generate_observations.py](./example02_generate_observations.py).

The Hamiltonian for this problem is:

$$
H = \int dx \left\[\frac{1}{2}D \left(\frac{d\phi}{dx}\right)^2 + \frac{1}{4}\kappa\phi^4
+ \phi f\right].
$$

We introduce model error in a continuous way using a parameter $\gamma$ that
ranges from $0$ to $1$.
When $\gamma$ is $0$ the model is wrong. When $\gamma$ is $1$ the model is correct.

We do this in two different ways:

### Example 2.a: Model error in the source term

In this example we make the source term:

$$
f(x;\gamma) = \gamma \cos(4x) + (1-\gamma)e^{-x},
$$

where $\gamma$ is fixed at different levels from $0$ to $1$.

### Example 2.b: Model error in the energy

In this example we use the correct source term, but we change the Hamiltonian to:

$$
H = \int dx \left\[\frac{1}{2}D \left(\frac{d\phi}{dx}\right)^2 + \gamma\frac{1}{4}\kappa\phi^4
+ (1-\gamma)\frac{1}{2}\phi^2
+ \phi f\right].
$$

## Running the examples

Make sure you have compiled the code following the instructions 
[here](../README.md).
The script [example02_run.sh](./example02_run.sh) reproduces the paper figures.
To run it, change in the directory `./examples` and type in your terminal:
```
./example02_run.sh
```

If you wish to change any of default settings, feel free to edit the 
corresponding configuration files:
+ [example02a.yml](./example02a.yml) for Example 2.a, and
+ [example02b.yml](./example02b.yml) for Example 2.b.

## The results

The above script creates the following figures and puts them in directories
called `example02a_results` and `example02b_results` for Example 2.a and 2.b,
respectively.

### Example 2.a

![Example 2.a](./paper_figures/example02a.png)

### Example 2.b

![Example 2.b](./paper_figures/example02b.png)
