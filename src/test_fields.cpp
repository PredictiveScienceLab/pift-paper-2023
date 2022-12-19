/* Unitests for the field classes.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/19/2022
 *
 */

#include "domain.hpp"
#include "fourier.hpp"
#include "utils.hpp"
#include "io.hpp"

#include <random>

using namespace std;


int main(int argc, char* argv[]) {
  mt19937 rng;

  const int num_terms = 4;
  const float bounds[2] = {0.0, 1.0};

  using F = float;
  using Domain = UniformRectangularDomain<F, mt19937>;
  using Field = Fourier1DField<F, Domain>;
  using CField = Constrained1DField<F, Field, Domain>;

  Domain domain(bounds, 1, rng);
  Field psi(domain, num_terms);

  const F bndry_values[2] = {0.0, 0.0};
  CField phi(psi, domain, bndry_values);

  const int n = 100;
  F x[n];
  F w[phi.get_dim_w()];
  normal_distribution<F> norm(0, 1);

  // Some points to evaluate things on
  linspace(psi.a(), psi.b(), n, x);
  savetxt(x, n, "src/unit_test_x.csv");

  // Some random weights
  generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});
  savetxt(w, psi.get_dim_w(), "src/unit_test_w.csv");

  // Evaluate the parameterization and its gradients
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
  savetxt(prolong, n, psi.get_prolong_size(), "src/unit_test_psi.csv");
  savetxt(grad_w_prolong, n, psi.get_grad_w_prolong_size(),
      "src/unit_test_grad_w_psi.csv");

  // Evaluate the constrained parameterization and its gradients
  for(int i=0; i<n; i++) {
    phi(
        x + i, 
        w, 
        prolong + phi.get_prolong_size() * i, 
        grad_w_prolong + phi.get_grad_w_prolong_size() * i
    );
  }
  savetxt(prolong, n, psi.get_prolong_size(), "src/unit_test_phi.csv");
  savetxt(grad_w_prolong, n, phi.get_grad_w_prolong_size(),
      "src/unit_test_grad_w_phi.csv");

  return 0;
}

