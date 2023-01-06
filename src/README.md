# API Documentation

## Code breakdown
The code code is broken down in the following C++ header files implementing
several template classes. For the details, we suggest that you read the
comments in the files.

A short description of each file is:
+ [pift.hpp](./pift.hpp): The header file that includes all other header files.
+ [utils.hpp](./utils.hpp): Some utility template functions.
+ [io.hpp](./io.hpp): Template functions that facilitate input/output.
+ [domain.hpp](./domain.hpp): Template class representing a spatial domain.
+ [field.hpp](./field.hpp): Field template classes.
+ [fourier.hpp](./fourier.hpp): A field template class that uses a Fourier basis.
+ [hamiltonian.hpp](./hamiltonian.hpp): Abstract template class for Hamiltonians
and some template classes for constructing unbiased estimators of various quantities.
+ [likelihood.hpp](./likelihood.hpp): Template classes related to likelihoods.
+ [posterior.hpp](./posterior.hpp): Template classes related to posteriors.
+ [sgld.hpp](./sgld.hpp): Template function implementation of stochastic gradient
Langevin dynamics.

## Using the code
The code resides in the namespace `pift`. You just need to include the header
file [pift.hpp](./pift.hpp) to start using it.
