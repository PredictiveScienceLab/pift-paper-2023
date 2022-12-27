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
#include <cstdio>
#include <cassert>

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
using UEGradWH = pift::UEIntegralGradWH<F, H, CFField, Domain>;
using UEGradWL = pift::UEGradWL<F, L, RNG>;
using UEGradWP = pift::UEGradWPost<F, UEGradWH, UEGradWL>;

int main(int argc, char* argv[]) {
  if(argc != 3) {
    std::cerr << "Usage:\n\t" << argv[0] << " <BETA> <SIGMA>" << std::endl;
    exit(1);
  }

  // The beta we want to use
  const F beta = static_cast<F>(std::stod(argv[1]));
  assert(beta > 0.0);
  const F sigma = static_cast<F>(std::stod(argv[2]));
  assert(sigma > 0.0);

  char parsed[256];
  snprintf(parsed, 256, "beta=%.2e_sigma=%.2e", beta, sigma);

  // Prefix for output files
  std::string prefix = "posterior_" + std::string(parsed);

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
  UEGradWP eu_p(eu_h, eu_l);

  // Some random weights to get us started
  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0, 1 / std::sqrt(num_terms));
  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});

  // Sample from the posterior of the w's
  std::string warmup_out_file = prefix + "_warmup.csv";
  std::string samples_out_file = prefix + "_samples.csv";
  pift::SGLDParams<F> sgld_params;
  sgld_params.alpha = 1e-2/beta;
  sgld_params.beta = 0.0;
  sgld_params.gamma = 0.51;
  sgld_params.save_freq = 10'000;
  sgld_params.disp = true;
  sgld_params.disp_freq = 1'000'000;
  const int num_warmup = 20'000'000;
  const int num_samples = 20'000'000;
  sgld_params.save_to_file = true;
  sgld_params.out_file = warmup_out_file;
  sgld(eu_p, w, phi.get_dim_w(), rng, num_warmup, sgld_params);
  sgld_params.init_it = num_warmup;
  sgld_params.save_to_file = true;
  sgld_params.out_file = samples_out_file;
  sgld(eu_p, w, phi.get_dim_w(), rng, num_samples, sgld_params);

  // Postprocess the results
  const int n = 100;
  postprocess<F>(phi, domain, n, samples_out_file, prefix);

  // Save also the true solution
  //
  F x[n];
  F y[n];
  pift::linspace(domain.a(0), domain.b(0), n, x);
  std::transform(x, x + n, y, solution);
  pift::savetxt(y, n, prefix + "_true.csv");

  return 0;
}

