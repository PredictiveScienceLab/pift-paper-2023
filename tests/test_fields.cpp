/* Unitests for the field classes.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/19/2022
 *
 */

#include <random>

#include "pift.hpp"

int main(int argc, char* argv[]) {

  // Some definitions to make things easier
  using RNG = std::mt19937;
  using F = float;
  using Domain = pift::UniformRectangularDomain<F, RNG>;
  using FField = pift::Fourier1DField<F, Domain>;
  using CFField = pift::Constrained1DField<F, FField, Domain>;

  // A random number generator
  RNG rng;

  // Number of Fourier terms
  const int num_terms = 4;

  // Spatial domain is [0, 1]
  const float bounds[2] = {0.0, 1.0};
  Domain domain(bounds, 1, rng);
  
  // A Fourier expansion on [0, 1]
  FField psi(domain, num_terms);

  // We want the function to be constrained to be zero at the boudnaries:
  const F bndry_values[2] = {0.0, 0.0};
  CFField phi(psi, domain, bndry_values);

  // Some points on which to evaluate the field
  const int n = 100;
  F x[n];
  pift::linspace(domain.a(0), domain.b(0), n, x);
  pift::savetxt(x, n, "tests/unit_test_x.csv");

  // Some random weights
  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0, 1);
  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});
  pift::savetxt(w, psi.get_dim_w(), "tests/unit_test_w.csv");

  // Evaluate the parameterization and its gradients, aka the prolongation.
  F prolong[psi.get_prolong_size() * n];
  F grad_w_prolong[psi.get_grad_w_prolong_size() * n];
  for(int i=0; i<n; i++) {
    psi(
        x + i, 
        w, 
        prolong + psi.get_prolong_size() * i, 
        grad_w_prolong + psi.get_grad_w_prolong_size() * i
    );
  }
  pift::savetxt(prolong, n, psi.get_prolong_size(), "tests/unit_test_psi.csv");
  pift::savetxt(grad_w_prolong, n, psi.get_grad_w_prolong_size(),
      "tests/unit_test_grad_w_psi.csv");

  // Evaluate the constrained parameterization and its gradients
  for(int i=0; i<n; i++) {
    phi(
        x + i, 
        w, 
        prolong + phi.get_prolong_size() * i, 
        grad_w_prolong + phi.get_grad_w_prolong_size() * i
    );
  }
  pift::savetxt(prolong, n, psi.get_prolong_size(), "tests/unit_test_phi.csv");
  pift::savetxt(grad_w_prolong, n, phi.get_grad_w_prolong_size(),
      "tests/unit_test_grad_w_phi.csv");

  return 0;
}

