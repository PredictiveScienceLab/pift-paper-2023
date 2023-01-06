# Physics-informed Information Field Theory

This repository replicates the result of the paper **ADD DOI WHEN PUBLISHED**.

## Installing the code

The main code is written in C++.
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
+ Several unit tests which are put in the directory `./tests`.

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
