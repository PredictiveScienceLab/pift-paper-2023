// Unitests for the Hamiltonian template classes.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022
//

#include <random>

#include "pift.hpp"
#include "diffusion.hpp"


int main(int argc, char* argv[]) {

  // Definition of some types
  using RNG = std::mt19937;
  using F = float;
  using Domain = pift::UniformRectangularDomain<F, RNG>;
  using FField = pift::Fourier1DField<F, Domain>;
  using CFField = pift::Constrained1DField<F, FField, Domain>;
  using H = Example1Hamiltonian<F>;
  using UEGradWH = pift::UEGradHAtFixedTheta<F, H, CFField, Domain>;

  // A random number generator
  RNG rng;

  // The spatial domain
  const F bounds[2] = {0.0, 1.0};
  Domain domain(bounds, 1, rng);

  // The Fourier parameterization for the field
  const int num_terms = 4;
  FField psi(domain, num_terms);

  // Constrain the field to satisfy the boundary conditions
  const F bndry_values[2] = {1.0, 1.0/10.0};
  CFField phi(psi, domain, bndry_values);

  // The beta we want to use
  F beta = 10'000.0;

  // The Hamiltonian
  H h(beta);

  // The number of collocation points to use at each iteration
  int num_collocation = 1;

  // An unbiased estimator of the gradient of the Hamiltonian with respect
  // to w
  UEGradWH eu_h(h, phi, domain, num_collocation, nullptr);

  // Some points on which to evaluate things
  const int n = 100;
  F x[n];
  pift::linspace(phi.a(), phi.b(), n, x);
  pift::savetxt(x, n, "tests/unit_test_x.csv");

  // Some random weights to get us started
  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0, 1);

  // Some random weights
  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});
  pift::savetxt(w, psi.get_dim_w(), "tests/unit_test_w.csv");

  // Evaluate the Hamiltonian on these points
  F h_vals[n];
  F grad_w_h[n * phi.get_dim_w()];
  std::fill(grad_w_h, grad_w_h + n * phi.get_dim_w(), 0.0);
  for(int i=0; i<n; i++)
    h_vals[i] = eu_h.add_grad(x + i, w, grad_w_h + phi.get_dim_w() * i);

  pift::savetxt(h_vals, n, "tests/unit_test_H.csv");
  pift::savetxt(grad_w_h, n, phi.get_dim_w(), "tests/unit_test_grad_w_H.csv");

  // Now let's estimate the parameters using sgld
  pift::SGLDParams<F> sgld_params;
  sgld_params.alpha = 0.1/beta;
  sgld_params.beta = 0.0;
  sgld_params.gamma = 0.51;
  sgld_params.out_file = "tests/unit_test_field_prior.csv";
  sgld_params.save_freq = 10'000;
  sgld_params.disp = true;
  sgld_params.disp_freq = 100'000;
  const int num_warmup = 10'000'000;
  const int num_samples = 10'000'000;
  sgld_params.save_to_file = false;
  sgld(eu_h, w, phi.get_dim_w(), rng, num_warmup, sgld_params);
  sgld_params.init_it = num_warmup;
  sgld_params.save_to_file = true;
  sgld(eu_h, w, phi.get_dim_w(), rng, num_samples, sgld_params);

  return 0;
}

