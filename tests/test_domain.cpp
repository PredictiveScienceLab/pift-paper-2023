// Some unit tests for the UniformRectangularDomain template class.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022

#include <iostream>
#include <iomanip>

#include "pift.hpp"

// The type of floating point numbers
using F = float;
// The type of random generator we are using
using RNG = std::mt19937;
// Concrete type for the UniformRectangularDomain template
using URD = pift::UniformRectangularDomain<F, RNG>;

int main(int argc, char* argv[]) {
  RNG rng;

  std::cout.precision(3);

  // 1D interval test. Using [-1, 3].
  std::cout << "1D domain test" << std::endl;
  F bounds[2] = {-1.0, 3.0};
  URD domain_1d(bounds, 1, rng);
  F x[1];
  for(int i=0; i<10; i++) {
    domain_1d.sample(x);
    std::cout << i << " " << x[0] << std::endl;
  }

  // 2D interval. Using [-1, 3] x [-5, 6].
  std::cout << std::endl << "2D domain test" << std::endl;
  F bounds2[4] = {-1.0, 3.0, -5.0, 6.0};
  URD domain_2d(bounds2, 2, rng); 
  F y[2];
  for(int i=0; i<10; i++) {
    domain_2d.sample(y);
    std::cout << i << " (" << y[0] << ", " << y[1] << ")" << std::endl;
  }

  return 0;
}
