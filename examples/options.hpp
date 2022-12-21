// Some helper functions to facilitate reading the configuration parameters.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/21/2022
//

#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

template<typename T>
using VectorOfVectors = std::vector<std::vector<T>>;

// Structures representing the configuration file
template<typename T>
struct OutputConfig {
  std::string prefix;
}; // Output

template<typename T>
struct DomainConfig {
  VectorOfVectors<T> bounds;
}; // Domain config

template<typename T>
struct FieldConfig {
  int num_terms;
  std::vector<T> boundary_values;
}; // Field

template<typename T>
struct SGLDConfig {
  int num_collocation;
  T alpha;
  T alpha_times_beta;
  T beta;
  T gamma;
  T sigma_w;
  bool save_warmup;
  bool save_samples;
  int save_freq;
  bool disp;
  disp_freq: 100000
  # The number of warmup steps
  num_warmup: 10000000
  # The number of production samples
  num_samples: 10000000

}; // SGLDConfig

// A structure representing the configuration file
template<typename T>
struct Configuration {
  Output<T> output;
  DomainConfig<T> domain;
  FieldConfig<T> field;
}; // Configuration

// Reads the options from the command line.
void read_options(int argc, char* argv[], const std::string config_file) {

} // read_options
