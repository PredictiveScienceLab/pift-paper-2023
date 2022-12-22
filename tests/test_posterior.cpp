// Unitests for posterior sampling.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022
//

#include <random>
#include <algorithm>

#include "pift.hpp"
#include "example01.hpp"
#include "postprocessing.hpp"

// Definition of some types
using RNG = std::mt19937;
using F = float;
using Domain = pift::UniformRectangularDomain<F, RNG>;
using FField = pift::Fourier1DField<F, Domain>;
using CFField = pift::Constrained1DField<F, FField, Domain>;
using H = Example01Hamiltonian<F>;
using L = pift::GaussianLikelihood<F, CFField>;
using UEGradWH = pift::UEGradHAtFixedTheta<F, H, CFField, Domain>;
using UEGradWL = pift::UEGradWLAtFixedTheta<F, L, RNG>;
using UEGradP = pift::UEGradWPostAtFixedTheta<F, UEGradWH, UEGradWL>;

int main(int argc, char* argv[]) {
  // Prefix for output files
  std::string prefix = "posterior";

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
  F beta = 1.0;

  // The Hamiltonian
  H h(beta);

  // An unbiased estimator of the Hamiltonian
  int num_collocation = 1;
  UEGradWH eu_h(h, phi, domain, num_collocation, nullptr);

  // Some random data
  const int num_obs = 10;
  F x_obs[num_obs];
  F y_obs[num_obs];
  // The solution function
  auto solution = h.get_solution(bndry_values);
  // The measurement function
  const F sigma = 0.01;
  std::normal_distribution<F> norm_obs(0, sigma);
  auto measure = [&solution, &norm_obs, &rng](const F& x) {
    return solution(x) + norm_obs(rng);
  };
  // Sample the x observations
  domain.sample(x_obs, num_obs);
  pift::savetxt(x_obs, num_obs, prefix + "_x_obs.csv");
  // Sample the y observations
  std::transform(x_obs, x_obs + num_obs, y_obs, measure);
  pift::savetxt(y_obs, num_obs, prefix + "_y_obs.csv");

  // The likelihood
  L l(phi, num_obs, x_obs, y_obs, sigma);

  // An unbiased estimator of the likelihood
  const int batch_size = 1;
  UEGradWL eu_l(l, nullptr, batch_size, rng);

  // An unbiased estimator of minus log the posterior
  UEGradP eu_p(eu_h, eu_l);

  // Some random weights to get us started
  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0, 1);
  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});

  // Sample from the posterior of the w's
  pift::SGLDParams<F> sgld_params;
  sgld_params.alpha = 0.001;
  sgld_params.beta = 0.0;
  sgld_params.gamma = 0.51;
  std::string samples_out_file = prefix + "_w.csv";
  sgld_params.out_file = samples_out_file;
  sgld_params.save_freq = 10'000;
  sgld_params.disp = true;
  sgld_params.disp_freq = 1'000'000;
  const int num_warmup = 10'000'000;
  const int num_samples = 10'000'000;
  sgld_params.save_to_file = false;
  sgld(eu_p, w, phi.get_dim_w(), rng, num_warmup, sgld_params);
  sgld_params.init_it = num_warmup;
  sgld_params.save_to_file = true;
  sgld(eu_p, w, phi.get_dim_w(), rng, num_samples, sgld_params);

  // Postprocess the results
  postprocess<F>(phi, domain, 100, samples_out_file, prefix);

  return 0;
}

