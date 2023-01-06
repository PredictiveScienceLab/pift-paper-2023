// Example 03 of the paper.
//
// Authors:
//  Ilias Bilionis
//
// Date:
//  1/3/2023
//

#include <random>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "pift.hpp"
// The Hamiltonian we will use is here:
#include "example03.hpp"

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
using H = Example03Hamiltonian<F, FField>;
// Type for likelihood
using L = pift::GaussianLikelihood<F, CFField>;
// Type for the unbiased estimator of the integral of the gradient of the
// Hamiltonian with respect to w
using UEGradWH = pift::UEIntegralGradWH<F, H, CFField, Domain>;
// Type for the unbiased estimator of the integral of the gradient of the
// Hamiltonian with respect to theta
using UEGradThetaH = pift::UEIntegralGradThetaH<F, H, CFField, Domain>;
// Type for unbiased estimator for minus the grad w of the log likelihood
using UEGradWL = pift::UEGradWL<F, L, RNG>;
// Type for unbiased estimator for minus the grad w of the log posterior
using UEGradWP = pift::UEGradWPost<F, UEGradWH, UEGradWL>;
// Type for unbiased estimator for the prior expectation of grad theta of the
// Hamiltonian
using UEGradThetaPrior = pift::UEGradThetaHF<F, UEGradWH, UEGradThetaH, RNG>;
// Type for unbiased estimator for the posterior expectation of grad theta of the
// Hamiltonian
using UEGradThetaPost = pift::UEGradThetaHF<F, UEGradWP, UEGradThetaH, RNG>;
// Type for the unbiased estimator of grad theta of minus the log posterior
using UEGradThetaMLP = 
  pift::UEGradThetaMinusLogPost<F,UEGradThetaPrior,UEGradThetaPost>;

int main(int argc, char* argv[]) {
  if(argc != 6) {
    std::cerr << "Usage:\n\t" << argv[0] 
        << " <BETA> <CONFIG_FILE> <N> <SIGMA> <ID>" << std::endl;
    exit(1);
  }

  // Read beta from the command line
  const F beta = static_cast<F>(std::stod(argv[1]));
  assert(beta > 0.0);
  const int num_obs = std::stoi(argv[3]);
  assert(num_obs >= 1);
  const F sigma = static_cast<F>(std::stod(argv[4]));
  assert(sigma >= 0);
  const int id = std::stoi(argv[5]);
  assert(id >= 0);
  char buffer[256];
  snprintf(buffer, 256, "n=%d_sigma=%1.2e_%d",num_obs, sigma, id);
  // Open the configuration file to read the rest of the parameters
  std::string config_file = argv[2];
  //std::filesystem::path config_file_full_path(config_file);
  if (not std::filesystem::exists(config_file)) {
    std::cerr << "Configuration file `" << config_file << "` was not found."
              << std::endl;
    exit(2);
  }
    
  YAML::Node yaml = YAML::LoadFile(config_file); 
  Configuration03<F> config(yaml);

  // The data files
  std::string x_file = "example02_" + std::string(buffer) 
    + "_x_obs.csv";
  std::string y_file = "example02_" + std::string(buffer) 
    + "_y_obs.csv";

  // The output prefix
  char buffer2[256];
  snprintf(buffer2, 256, "beta=%1.2e_%s", beta, buffer);
  std::string prefix = config.output.prefix + "_" + std::string(buffer2);

  // A random number generator
  auto const hes = std::random_device{}();
  RNG rng(hes);

  Domain domain(config.domain.bounds, rng);

  // Make the spatial domain
  FField psi(domain, config.field.num_terms);

  // Constrain the field to satisfy the boundary conditions
  CFField phi(psi, domain, config.field.boundary_values);

  // The field representing the source term
  FField f(domain, config.source.num_terms);

  // The Hamiltonian
  H h(beta, f, config.source.precision);

  // Initialize the parameters
  std::normal_distribution<F> norm(0, 1);
  F theta[h.get_num_params()];
  for(int i=0; i<h.get_num_params(); i++)
    theta[i] = config.parameters.init_mean[i] +
               config.parameters.init_std[i] * norm(rng);


  // The unbiased estimator of the integral of the gradient of the
  // Hamiltonian with respect to theta
  UEGradThetaH ue_int_grad_theta_H(h, phi, domain, config.parameters.num_collocation);

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
  auto theta_prior_params = config.parameters.prior.get_theta_params();
  theta_prior_params.sgld_params.out_file = prefix + "_prior_ws.csv";
  UEGradThetaPrior ue_prior_exp_int_grad_theta_H(
      ue_grad_w_h,
      ue_int_grad_theta_H,
      rng,
      theta_prior_params
  );


  // Load the filenames
  auto x_obs = pift::loadtxtvec<F>(x_file);
  auto y_obs = pift::loadtxtvec<F>(y_file);
  assert(x_obs.size() == y_obs.size());
  L l(phi, x_obs.size(), x_obs.data(), y_obs.data(), sigma);
  UEGradWL ue_grad_w_l(l, theta, config.parameters.post.batch_size, rng);
  UEGradWH ue_grad_w_h_tmp(
      h,
      phi,
      domain,
      config.parameters.post.num_collocation,
      theta
  );

  UEGradWP ue_grad_w_post(ue_grad_w_h_tmp, ue_grad_w_l);

  // Unbiased estimator of the posterior expectation of the integral of
  // grad theta of the Hamiltonian
  auto theta_post_params = config.parameters.post.get_theta_params();
  theta_post_params.sgld_params.out_file = prefix + "_post_ws.csv";
  UEGradThetaPost ue_post_exp_int_grad_theta_H(
       ue_grad_w_post,
       ue_int_grad_theta_H,
       rng,
       theta_post_params
  ); 

  // Unbiased estimator of grad theta of minus log posterior
  UEGradThetaMLP ue_grad_theta(
      ue_prior_exp_int_grad_theta_H,
      ue_post_exp_int_grad_theta_H
  );


  ue_grad_theta.warmup(theta);
  auto sgld_params = config.parameters.sgld.get_sgld_params();
  sgld_params.alpha /= beta;
  sgld_params.out_file = prefix + "_theta.csv";
  sgld(
      ue_grad_theta,
      theta,
      h.get_num_params(),
      rng,
      config.parameters.sgld.num_warmup,
      sgld_params,
      false
  );

  // Postprocess the results
  auto prior_prefix = prefix + "_prior";
  postprocess<F>(
      phi, domain, config.postprocess.num_points_per_dim[0],
      theta_prior_params.sgld_params.out_file,
      prior_prefix
  );
  auto post_prefix = prefix + "_post";
  postprocess<F>(
      phi, domain, config.postprocess.num_points_per_dim[0],
      theta_post_params.sgld_params.out_file,
      post_prefix
  );
  postprocess_source<F>(
      f, domain, config.postprocess.num_points_per_dim[0],
      sgld_params.out_file,
      prefix
  );

  return 0;
} // main
