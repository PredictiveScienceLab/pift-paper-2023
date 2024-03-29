# Physics-informed Information Field Theory

This repository replicates the result of the paper (https://doi.org/10.1016/j.jcp.2023.112100).

## Code outline

The code is structured as follows:
+ [src](./src): Contains the C++ code that implements the methodology.
Note that for the time being we only support fields with one input and one
output. Support for multiple outputs will be added in the future.
+ [examples](./examples): Contains the C++ code that implements the paper
examples, and the Python scripts that make the plots.
+ [tests](./tests): Contains some unit tests. These may not be up to date.

## Installing the code

The code for Examples 1-3 is written in C++.
The plotting code is written in Python.

The requirements for the C++ code is:
+ [YAML-CPP](https://github.com/jbeder/yaml-cpp) for reading YAML configuration
  files. If you are on OS X and you are using
  [homebrew](https://brew.sh), then you can simply do:
  ```
    brew install yaml-cpp
  ```
  If you are on a different OS, you are on your own.
  In any case, once you have YAML-CPP installed, you need to edit the
  [makefile](./makefile) to make the variable `YAMLCPP` point to the right
  folder.
+ [GNU Scientific Library](https://www.gnu.org/software/gsl/). This is only
  used in [Example 3b](examples/example03.md) to find eigenvalues of a
  symmetric matrix. Again, if you are using on OS X and using 
  [homebrew](https://brew.sh), then you do:
  ```
    brew install gsl
  ```

To compile the C++ code, simply run:
```
make all
```
in the first directory of the problem.
This command will compile the following executables:
+ `examples/example01`: Example 1 of the paper.
+ `examples/example02a`: Example 2.a of the paper.
+ `examples/example02b`: Example 2.b of the paper.
+ `examples/example03a`: Example 3.a of the paper.
+ `examples/example03b`: Example 3.b of the paper.

The requirements for the Python plotting scripts are (ignoring standard libraries):
+ [matplotlib](https://matplotlib.org)
+ [seaborn](https://seaborn.pydata.org)
+ [texlive](https://tug.org/texlive/) If you are on OS X and using homebrew,
  then run
  ```
    brew install texlive
  ```
  If for some reason you cannot install texlive, you will need to manually edit
  the Python plotting scripts and comment out the lines:
  ```
    plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
    params = {'text.usetex' : True,
              'font.size' : 9,
              'font.family' : 'lmodern'
              }
  ```

The code for Example 4 is in a Jupyter notebook.
The requirements are:
+ [numpyro](https://num.pyro.ai/en/stable/getting_started.html)
+ [jax](https://github.com/google/jax)
+ [h5py](https://www.h5py.org)

## Reproducing the paper results

Follow the links to see how you can reproduce the paper results for each eaxmple:
+ [Example 1](./examples/example01.md)
+ [Example 2](./examples/example02.md)
+ [Example 3](./examples/example03.md)
+ [Example 4](./examples/2D-Allen-Cahn.ipynb)

**Note:** Because the random seed changes every time your run the code,
the figures your produce may be slightly different from the ones found in the
paper.
We dedided not to fix the random seed to avoid a positive results bias,
or ``hero'' runs.
