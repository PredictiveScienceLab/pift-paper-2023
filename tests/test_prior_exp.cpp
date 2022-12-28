// Tests the unbiased estimator of the prior expectation of the integral of
// grad theta of the Hamiltonian.
//
// Authors:
//  Ilias Bilionis
//
// Date:
//  12/28/2022

#include <random>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "pift.hpp"
#include "example02.hpp"

#include "options.hpp"
#include "postprocessing.hpp"

// Definition of some types
using RNG = std::mt19937;
using F = float;
using Domain = pift::UniformRectangularDomain<F, RNG>;
// Type for parameterized field
using FField = pift::Fourier1DField<F, Domain>;
// Type for constrained parameterized field
using CFField = pift::Constrained1DField<F, FField, Domain>;
// Type for Hamiltonian
using H = Example02Hamiltonian<F>;
// Type for likelihood
using L = pift::GaussianLikelihood<F, CFField>;
// Type for the unbiased estimator of the integral of the gradient of the
// Hamiltonian with respect to w
using UEGradWH = pift::UEIntegralGradWH<F, H, CFField, Domain>;
// Type for the unbiased estimator of the integral of the gradient of the
// Hamiltonian with respect to theta
using UEGradThetaH = pift::UEIntegralGradThetaH<F, H, CFField, Domain>;
// Type for unbiased estimator for the prior expectation of grad theta of the
// Hamiltonian
using UEGradThetaPrior = pift::UEGradThetaHF<F, UEGradWH, UEGradThetaH, RNG>;

int main(int argc, char* argv[]) {
  const F gamma = 1.0;
  char parsed_gamma[256];
  snprintf(parsed_gamma, 256, "gamma=%.2e", gamma);

  // Open the configuration file to read the rest of the parameters
  std::string config_file = "test_config.yml";
  //std::filesystem::path config_file_full_path(config_file);
  if (not std::filesystem::exists(config_file)) {
    std::cerr << "Configuration file `" << config_file << "` was not found."
              << std::endl;
    exit(2);
  }
    
  YAML::Node yaml = YAML::LoadFile(config_file); 
  Configuration02<F> config(yaml);

  // The output prefix
  std::string prefix = config.output.prefix + "_" + std::string(parsed_gamma);

  // A random number generator
  RNG rng;

  Domain domain(config.domain.bounds, rng);

  // Make the spatial domain
  FField psi(domain, config.field.num_terms);

  // Constrain the field to satisfy the boundary conditions
  CFField phi(psi, domain, config.field.boundary_values);

  // The Hamiltonian
  H h(gamma);

  // Initialize the parameters
  std::normal_distribution<F> norm(0, 1);
  F theta[h.get_num_params()];
  for(int i=0; i<h.get_num_params(); i++)
    theta[i] = config.parameters.init_mean[0] +
               config.parameters.init_std[0] * norm(rng);

  // The unbiased estimator of the integral of the gradient of the
  // Hamiltonian with respect to theta
  UEGradThetaH ue_int_grad_theta_H(h, phi, domain,
      config.parameters.prior.num_collocation);

  // Unbiased estimator used to take expectations over prior
  UEGradWH ue_grad_w_h(
      h,
      phi,
      domain,
      config.parameters.prior.num_collocation,
      theta
  );
  
  // Unbiased estimator of the prior expectation of the integral of grad theta
  // of the Hamiltonian
  auto theta_params = config.parameters.prior.get_theta_params();
  UEGradThetaPrior ue_prior_exp_int_grad_theta_H(
      ue_grad_w_h,
      ue_int_grad_theta_H,
      rng,
      theta_params
  );

  // Let's test this because we haven't really tested it
  F grad_theta[h.get_num_params()];
  ue_prior_exp_int_grad_theta_H.warmup(theta);

//  // An unbiased estimator of the gradient of the Hamiltonian with respect
//  // to w
//  UEGradWH eu_h(h, phi, domain, config.sgld.num_collocation, nullptr);
//
//  // Initialize the w's
//  F w[phi.get_dim_w()];
//  std::normal_distribution<F> norm(0, config.sgld.sigma_w);
//  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});
//
//  // Do the warmup
//  sgld_params.save_to_file = config.sgld.save_warmup;
//  sgld_params.out_file = warmup_out_file; 
//  sgld(eu_h, w, phi.get_dim_w(), rng, config.sgld.num_warmup, sgld_params);
//
//  // Do the production samples
//  sgld_params.init_it = config.sgld.num_warmup;
//  sgld_params.save_to_file = config.sgld.save_samples;
//  sgld_params.out_file = samples_out_file;
//  sgld(eu_h, w, phi.get_dim_w(), rng, config.sgld.num_samples, sgld_params);
//
//  // Postprocess the results
//  const int n = config.postprocess.num_points_per_dim[0];
//  postprocess<F>(
//      phi, domain, config.postprocess.num_points_per_dim[0],
//      samples_out_file,
//      prefix
//  );
//
//  // We are done
//  std::cout << "*** Done ***" << std::endl;
//  std::cout << "I wrote the following files:" << std::endl;
//  if(config.sgld.save_warmup)
//    std::cout << "\t- " << warmup_out_file << std::endl;
//  if(config.sgld.save_samples)
//    std::cout << "\t- " << samples_out_file << std::endl;
//  std::cout << "\t- " << prefix + "_x.csv" << std::endl;
//  std::cout << "\t- " << prefix + "_phi.csv" << std::endl;

  return 0;
} // main
