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
