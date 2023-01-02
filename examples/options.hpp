// Some helper functions to facilitate reading the configuration parameters.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/21/2022
//

#ifndef EXAMPLES_OPTIONS_HPP
#define EXAMPLES_OPTIONS_HPP

#include <vector>
#include <string>
#include <cassert>
#include <yaml-cpp/yaml.h>

#include "pift.hpp"

template<typename T>
using VectorOfVectors2 = std::vector<std::vector<T>>;

// Structures representing the configuration file
struct OutputConfig {
  std::string prefix;
  OutputConfig(const YAML::Node& yaml) : 
    prefix(yaml["prefix"].as<std::string>())
  {}
}; // Output

template<typename T>
struct DomainConfig {
  VectorOfVectors2<T> bounds;
  DomainConfig(const YAML::Node& yaml) : 
    bounds(yaml["bounds"].as<VectorOfVectors2<T>>())
  {
    assert(bounds.size() != 0);
    for(int i=0; i<bounds.size(); i++) {
      assert(bounds[i].size() == 2);
      assert(bounds[i][0] < bounds[i][1]);
    }
  }
}; // Domain config

template<typename T>
struct FieldConfig {
  int num_terms;
  std::vector<T> boundary_values;
  FieldConfig(const YAML::Node& yaml) :
    num_terms(yaml["num_terms"].as<int>()),
    boundary_values(yaml["boundary_values"].as<std::vector<T>>())
  {
    assert(num_terms > 0);
  }
}; // Field

template<typename T>
struct SGLDConfig {
  T alpha;
  T beta;
  T gamma;
  bool save_to_file;
  int save_freq;
  bool disp;
  int disp_freq;
  int num_warmup;
  int num_samples;
  SGLDConfig(const YAML::Node& yaml) :
    alpha(yaml["alpha"].as<T>()),
    beta(yaml["alpha"].as<T>()),
    gamma(yaml["gamma"].as<T>()),
    save_to_file(yaml["save_to_file"].as<bool>()),
    save_freq(yaml["save_freq"].as<int>()),
    disp(yaml["disp"].as<bool>()),
    disp_freq(yaml["disp_freq"].as<int>()),
    num_warmup(yaml["num_warmup"].as<int>()),
    num_samples(yaml["num_samples"].as<int>())
  {
    assert(alpha > 0.0);
    assert(beta >= 0.0);
    assert(gamma > 0.5 && gamma <= 1.0);
    assert(save_freq >= 1);
    assert(disp_freq >= 1);
    assert(num_warmup >= 0);
    assert(num_samples >= 0);
  }

  pift::SGLDParams<T> get_sgld_params() const {
    pift::SGLDParams<T> sgld_params;
    sgld_params.alpha = alpha;
    sgld_params.beta = beta;
    sgld_params.gamma = gamma;
    sgld_params.save_to_file = save_to_file;
    sgld_params.save_freq = save_freq;
    sgld_params.disp = disp;
    sgld_params.disp_freq = disp_freq;
    return sgld_params;
  }
}; // SGLDConfig

template<typename T>
struct PriorConfig {
  int num_chains;
  int num_init_warmup;
  int num_bursts;
  bool reinitialize_ws;
  int num_collocation;
  T sigma_w;
  bool adjust_alpha;
  SGLDConfig<T> sgld;

  PriorConfig(const YAML::Node& yaml) :
    num_chains(yaml["num_chains"].as<int>()),
    num_init_warmup(yaml["num_init_warmup"].as<int>()),
    num_bursts(yaml["num_bursts"].as<int>()),
    reinitialize_ws(yaml["reinitialize_ws"].as<bool>()),
    num_collocation(yaml["num_collocation"].as<int>()),
    sigma_w(yaml["sigma_w"].as<T>()),
    adjust_alpha(yaml["adjust_alpha"].as<bool>()),
    sgld(yaml["sgld"])
  {
    assert(num_chains >= 1);
    assert(num_init_warmup >= 0);
    assert(num_bursts >= 1);
    assert(num_collocation >= 1);
  }

  pift::UEThetaParams<T> get_theta_params() const {
    pift::UEThetaParams<T> theta_params;
    theta_params.num_chains = num_chains;
    theta_params.num_init_warmup = num_init_warmup;
    theta_params.num_per_it_warmup = sgld.num_warmup;
    theta_params.num_bursts = num_bursts;
    theta_params.num_thinning = sgld.num_samples;
    theta_params.init_w_sigma = sigma_w;
    theta_params.adjust_alpha = adjust_alpha;
    theta_params.reinitialize_ws = reinitialize_ws;
    theta_params.sgld_params = sgld.get_sgld_params();
    return theta_params;
  }
}; // PriorConfig

template<typename T>
struct PostConfig : public PriorConfig<T> {
  int batch_size;

  PostConfig(const YAML::Node& yaml) :
    PriorConfig<T>(yaml),
    batch_size(yaml["batch_size"].as<int>())
  {
    assert(batch_size >= 1);
  }
}; // PostConfig

template<typename T>
struct ParamsConfig {
  std::vector<T> init_mean;
  std::vector<T> init_std;
  int num_collocation;
  SGLDConfig<T> sgld;
  PriorConfig<T> prior;
  PostConfig<T> post;

  ParamsConfig(const YAML::Node& yaml) :
    init_mean(yaml["init_mean"].as<std::vector<T>>()),
    init_std(yaml["init_std"].as<std::vector<T>>()),
    num_collocation(yaml["num_collocation"].as<int>()),
    sgld(yaml["sgld"]),
    prior(yaml["prior"]),
    post(yaml["post"])
  {
    assert(init_mean.size() == init_std.size());
    for(int i=0; i<init_std.size(); i++)
      assert(init_std[i] >= 0.0);
  }
}; // ParamsConfig

struct PostProcessConfig {
  std::vector<int> num_points_per_dim;
  PostProcessConfig(const YAML::Node& yaml) :
    num_points_per_dim(yaml["num_points_per_dim"].as<std::vector<int>>())
  {
    for(int i=0; i<num_points_per_dim.size(); i++)
      assert(num_points_per_dim[i] > 0);
  }
}; // PostProcessConfig

// A structure representing the configuration file for example 1
template<typename T>
struct Configuration01 {
  OutputConfig output;
  DomainConfig<T> domain;
  FieldConfig<T> field;
  int num_collocation;
  T sigma_w;
  SGLDConfig<T> sgld;
  PostProcessConfig postprocess;

  Configuration01(const YAML::Node& yaml) :
    output(yaml["output"]),
    domain(yaml["domain"]),
    field(yaml["field"]),
    num_collocation(yaml["num_collocation"].as<int>()),
    sigma_w(yaml["sigma_w"].as<T>()),
    sgld(yaml["sgld"]),
    postprocess(yaml["postprocess"])
  {
    assert(domain.bounds.size() == postprocess.num_points_per_dim.size());
    assert(num_collocation >= 1);
    assert(sigma_w >= 0.0);
  }
}; // Configuration01

// A structure representing the configuration file for example 2
template<typename T>
struct Configuration02 {
  OutputConfig output;
  DomainConfig<T> domain;
  FieldConfig<T> field;
  ParamsConfig<T> parameters;
  PostProcessConfig postprocess;

  Configuration02(const YAML::Node& yaml) :
    output(yaml["output"]),
    domain(yaml["domain"]),
    field(yaml["field"]),
    parameters(yaml["parameters"]),
    postprocess(yaml["postprocess"])
  {
    assert(domain.bounds.size() == postprocess.num_points_per_dim.size());
  }
}; // Configuration02


#endif // EXAMPLES_OPTIONS
