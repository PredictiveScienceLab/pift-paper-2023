// Example 01 of the paper.
//
// Authors:
//  Ilias Bilionis
//  Alex Alberts
//
// Date:
//  12/20/2022

#include <random>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <yaml-cpp/yaml.h>

#include "pift.hpp"
// This is where you can find the Hamiltonian of the example:
#include "example01.hpp"

#include "options.hpp"
#include "postprocessing.hpp"

// Definition of some types
using RNG = std::mt19937;
using F = float;
using Domain = pift::UniformRectangularDomain<F, RNG>;
using FField = pift::Fourier1DField<F, Domain>;
using CFField = pift::Constrained1DField<F, FField, Domain>;
using H = Example01Hamiltonian<F>;
using UEGradWH = pift::UEGradHAtFixedTheta<F, H, CFField, Domain>;

int main(int argc, char* argv[]) {
  if(argc != 3) {
    std::cerr << "Usage:\n\t" << argv[0] 
        << " <BETA> <CONFIG_FILE>" << std::endl;
    exit(1);
  }

  // Read beta from the command line
  const F beta = static_cast<F>(std::stod(argv[1]));
  assert(beta >= 0.0);
  char parsed_beta[256];
  snprintf(parsed_beta, 256, "beta=%.2e", beta);

  // Open the config_fileuration file to read the rest of the parameters
  std::string config_file = argv[2];
  //std::filesystem::path config_file_full_path(config_file);
  if (not std::filesystem::exists(config_file)) {
    std::cerr << "Configuration file `" << config_file << "` was not found."
              << std::endl;
    exit(2);
  }
    
  YAML::Node yaml = YAML::LoadFile(config_file); 
  Configuration<F> config(yaml);

  // The output prefix
  std::string prefix = config.output.prefix + "_" + std::string(parsed_beta);

  // Read the optimization parameters
  // See example01.yml for what they mean
  pift::SGLDParams<F> sgld_params = config.sgld.get_sgld_params();
  sgld_params.alpha = config.sgld.alpha / beta;
  std::string warmup_out_file = prefix + "_warmup.csv";
  std::string samples_out_file = prefix + "_samples.csv";

  // A random number generator
  RNG rng;

  Domain domain(config.domain.bounds, rng);

  // Make the spatial domain
  FField psi(domain, config.field.num_terms);

  // Constrain the field to satisfy the boundary conditions
  CFField phi(psi, domain, config.field.boundary_values);

  // The Hamiltonian
  H h(beta);

  // An unbiased estimator of the gradient of the Hamiltonian with respect
  // to w
  UEGradWH eu_h(h, phi, domain, config.sgld.num_collocation, nullptr);

  // Initialize the w's
  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0, config.sgld.sigma_w);
  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});

  // Do the warmup
  sgld_params.save_to_file = config.sgld.save_warmup;
  sgld_params.out_file = warmup_out_file; 
  sgld(eu_h, w, phi.get_dim_w(), rng, config.sgld.num_warmup, sgld_params);

  // Do the production samples
  sgld_params.init_it = config.sgld.num_warmup;
  sgld_params.save_to_file = config.sgld.save_samples;
  sgld_params.out_file = samples_out_file;
  sgld(eu_h, w, phi.get_dim_w(), rng, config.sgld.num_samples, sgld_params);

  // Postprocess the results
  const int n = config.postprocess.num_points_per_dim[0];
  postprocess<F>(
      phi, domain, config.postprocess.num_points_per_dim[0],
      samples_out_file,
      prefix
  );

  // We are done
  std::cout << "*** Done ***" << std::endl;
  std::cout << "I wrote the following files:" << std::endl;
  if(config.sgld.save_warmup)
    std::cout << "\t- " << warmup_out_file << std::endl;
  if(config.sgld.save_samples)
    std::cout << "\t- " << samples_out_file << std::endl;
  std::cout << "\t- " << prefix + "_x.csv" << std::endl;
  std::cout << "\t- " << prefix + "_phi.csv" << std::endl;

  return 0;
} // main
