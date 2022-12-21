// Unitests for the Hamiltonian template classes.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022
//

#include "domain.hpp"
#include "fourier.hpp"
#include "utils.hpp"
#include "io.hpp"
#include "diffusion.hpp"
#include "sgld.hpp"

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
  using H = Example1Hamiltonian<F>;
  using UEGradWH = UEGradHAtFixedTheta<F, H, CField, Domain>;

  Domain domain(bounds, 1, rng);
  Field psi(domain, num_terms);

  const F bndry_values[2] = {1.0, 1.0/10.0};
  CField phi(psi, domain, bndry_values);

  F beta = 10'000.0;
  H h(beta);

  int num_collocation = 10;
  UEGradWH eu_h(h, phi, domain, num_collocation, NULL);

  const int n = 100;
  F x[n];
  F w[phi.get_dim_w()];
  normal_distribution<F> norm(0, 1);

  // Some points to evaluate things on
  linspace(phi.a(), phi.b(), n, x);
  savetxt(x, n, "src/unit_test_x.csv");

  // Some random weights
  generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});
  savetxt(w, psi.get_dim_w(), "src/unit_test_w.csv");

  // Evaluate the Hamiltonian at these points
  F h_vals[n];
  F grad_w_h[n * phi.get_dim_w()];
  fill(grad_w_h, grad_w_h + n * phi.get_dim_w(), 0.0);
  for(int i=0; i<n; i++)
    h_vals[i] = eu_h.add_grad(x + i, w, grad_w_h + phi.get_dim_w() * i);

  savetxt(h_vals, n, "src/unit_test_H.csv");
  savetxt(grad_w_h, n, phi.get_dim_w(), "src/unit_test_grad_w_H.csv");

  // Now let's estimate the parameters using sgld
  SGLDParams<F> sgld_params;
  sgld_params.alpha = 0.1/beta;
  sgld_params.beta = 0.0;
  sgld_params.gamma = 0.51;
  sgld_params.out_file = "src/foo_prior.csv";
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

