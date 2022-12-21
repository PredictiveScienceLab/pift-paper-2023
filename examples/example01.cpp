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
#include <yaml-cpp/yaml.h>

#include "pift.hpp"
#include "diffusion.hpp"

// Definition of some types
using RNG = std::mt19937;
using F = float;
using Domain = pift::UniformRectangularDomain<F, RNG>;
using FField = pift::Fourier1DField<F, Domain>;
using CFField = pift::Constrained1DField<F, FField, Domain>;
using H = Example1Hamiltonian<F>;
using UEGradWH = pift::UEGradHAtFixedTheta<F, H, CFField, Domain>;
using VectorOfVectors = std::vector<std::vector<F>>;

int main(int argc, char* argv[]) {
  if(argc != 2) {
    std::cout << "Usage:\n\t" << argv[0] << " <BETA>" << std::endl;
    std::cout << "*** Note that this program also reads options ";
    std::cout << "from the file example1.yml. ***" << std::endl;
    exit(1);
  }

  // Read beta from the command line
  const F beta = static_cast<F>(std::stod(argv[1]));
  assert(beta >= 0.0);
  char parsed_beta[256];
  sprintf(parsed_beta, "beta=%.2e", beta);

  // Open the configuration file to read the rest of the parameters
  YAML::Node config = YAML::LoadFile("examples/example01.yml"); 

  // The output prefix
  std::string prefix = config["output"]["prefix"].as<std::string>();
  prefix += "_" + std::string(parsed_beta);
  std::cout << prefix << std::endl;

  // Read the bounds of the spatial domain
  const VectorOfVectors bounds = config["domain"].as<VectorOfVectors>();

  // The Fourier parameterization for the field
  const int num_terms = config["field"]["num_terms"].as<int>();
  assert(num_terms > 0);

  // The boundary values
  const std::vector<F> bndry_values = 
    config["field"]["boundary_values"].as<std::vector<F>>();

  // The parameters for stochastic gradient descent
  // The number of collocation points to use at each iteration
  const int num_collocation = config["sgld"]["num_collocation"].as<int>();
  assert(num_collocation > 0);

  // Read the optimization parameters
  // See example01.yml for what they mean
  pift::SGLDParams<F> sgld_params;
  sgld_params.alpha = config["sgld"]["alpha_times_beta"].as<F>() / beta;
  assert(sgld_params.alpha > 0.0);
  sgld_params.beta = config["sgld"]["beta"].as<F>();
  assert(sgld_params.beta >= 0.0);
  sgld_params.gamma = config["sgld"]["gamma"].as<F>();
  const F sigma_w = config["sgld"]["sigma_w"].as<F>();
  assert(sgld_params.gamma >= 0.0);
  bool save_warmup = config["sgld"]["save_warmup"].as<bool>();
  bool save_samples = config["sgld"]["save_samples"].as<bool>();
  std::string warmup_out_file = prefix + "_warmup.csv";
  std::string samples_out_file = prefix + "_samples.csv";
  sgld_params.save_freq = config["sgld"]["save_freq"].as<int>();
  sgld_params.disp = config["sgld"]["disp"].as<bool>();
  sgld_params.disp_freq = config["sgld"]["disp_freq"].as<int>();
  const int num_warmup = config["sgld"]["num_warmup"].as<int>();
  const int num_samples = config["sgld"]["num_samples"].as<int>();

  // A random number generator
  RNG rng;

  Domain domain(bounds, rng);

  // Make the spatial domain
  FField psi(domain, num_terms);

  // Constrain the field to satisfy the boundary conditions
  CFField phi(psi, domain, bndry_values);

  // The Hamiltonian
  H h(beta);

  // An unbiased estimator of the gradient of the Hamiltonian with respect
  // to w
  UEGradWH eu_h(h, phi, domain, num_collocation, nullptr);

  // Initialize the w's
  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0, sigma_w);
  std::generate(w, w + psi.get_dim_w(), [&norm, &rng]() {return norm(rng);});

  // Do the warmup
  sgld_params.save_to_file = save_warmup;
  sgld_params.out_file = warmup_out_file; 
  sgld(eu_h, w, phi.get_dim_w(), rng, num_warmup, sgld_params);

  // Do the production samples
  sgld_params.init_it = num_warmup;
  sgld_params.save_to_file = save_samples;
  sgld_params.out_file = samples_out_file;
  sgld(eu_h, w, phi.get_dim_w(), rng, num_samples, sgld_params);

  // TODO: Post process the results. Save the field values on
  // a mesh for each one of the production samples

  // We are done
  std::cout << "*** Done ***" << std::endl;
  std::cout << "I wrote the following files:" << std::endl;
  if(save_warmup)
    std::cout << "\t- " << warmup_out_file << std::endl;
  if(save_samples)
    std::cout << "\t- " << samples_out_file << std::endl;

  return 0;
} // main
