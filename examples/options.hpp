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
  bool save_warmup;
  bool save_samples;
  int save_freq;
  bool disp;
  int disp_freq;
  int num_warmup;
  int num_samples;
  SGLDConfig(const YAML::Node& yaml) :
    alpha(yaml["alpha"].as<T>()),
    beta(yaml["alpha"].as<T>()),
    gamma(yaml["gamma"].as<T>()),
    save_warmup(yaml["save_warmup"].as<bool>()),
    save_samples(yaml["save_samples"].as<bool>()),
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
    sgld_params.save_to_file = save_warmup;
    sgld_params.save_freq = save_freq;
    sgld_params.disp = disp;
    sgld_params.disp_freq = disp_freq;
    return sgld_params;
  }
}; // SGLDConfig

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
#endif // EXAMPLES_OPTIONS
