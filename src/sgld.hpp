// Implementation of Stochastic Gradient Langevin Dynamics.
//
// Reference:
//  
//  Welling & Teh, 2011, Bayesian Learning via stochastic gradient Langevin
//  dynamics.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/16/2022
//

#ifndef PIFT_OPT_HPP
#define PIFT_OPT_HPP

namespace pift {

#include <random>
#include <string>
#include <cmath>
#include <fstream>

template<typename T>
struct SGLDParams {

  SGLDParams() :
    alpha(1e-6),
    beta(0.0),
    gamma(0.51),
    init_it(0),
    save_to_file(false),
    out_file("sgld_deafult_out.csv"),
    save_freq(10),
    disp(true),
    disp_freq(100),
    grad_cap(0.1)
  {}

  T alpha;
  T beta;
  T gamma;
  int init_it;
  bool save_to_file;
  std::string out_file;
  int save_freq;
  bool disp;
  int disp_freq;
  T grad_cap;
};

// SGLD implementation.
// It allocates its own memory.
template<typename T, typename UE, typename R>
inline void sgld(UE& ue_func,
          T* w,
          const int& dim,
          R& rng,
          const int num_samples,
          const SGLDParams<T>& params = SGLDParams<T>(),
          const bool& sgd_only=false
) {
  T grad_w_H[dim];
  std::normal_distribution<T> norm(0, 1);
  sgld(ue_func, w, dim, rng, num_samples, grad_w_H, norm, params, sgd_only);
}

template <typename T, typename UE, typename R, typename Dist>
inline void sgld(UE& ue_func,
          T* w,
          const int& dim,
          R& rng,
          const int num_samples,
          T* grad_w_H,
          Dist& norm,
          const SGLDParams<T>& params = SGLDParams<T>(),
          const bool& sgd_only=false
) {
  std::ofstream of;
  if (params.save_to_file)
    of.open(params.out_file);
  sgld(ue_func, w, dim, rng, num_samples, grad_w_H, norm, of, params, sgd_only);
  if (params.save_to_file)
    of.close();
}

// SGLD implementation that requires providing 
// memory space grad_w, a standard normal distribution object,
// and an open ofstream to write to.
template <typename T, typename UE, typename R, typename Dist>
void sgld(UE& ue_func,
          T* w,
          const int& dim,
          R& rng,
          const int num_samples,
          T* grad_w_H,
          Dist& norm,
          std::ofstream& of,
          const SGLDParams<T>& params = SGLDParams<T>(),
          const bool& sgd_only=false
) {
  const int max_it = num_samples + params.init_it;
  for(int it=params.init_it; it<max_it; it++) {
    const T epsilon = params.alpha 
      / std::pow((params.beta + it + 1), params.gamma);
    const T h_val = ue_func(w, grad_w_H);
    if(sgd_only) {
      for(int i=0; i<dim; i++)
        w[i] += -epsilon * grad_w_H[i];
    } else {
      T sqrt_2_epsilon = std::sqrt(2.0 * epsilon);
      for(int i=0; i<dim; i++) {
        T step = epsilon * grad_w_H[i];
        if(step > params.grad_cap)
          step = params.grad_cap;
        else if(step < -params.grad_cap)
          step = -params.grad_cap;
        w[i] += (-step + sqrt_2_epsilon * norm(rng));
      }
    }
    if (params.save_to_file && it % params.save_freq == 0) {
      of << h_val << " ";
      cout_vec(w, dim, of);
    }
    if (params.disp && it % params.disp_freq == 0)
      cout_vec(w, dim, std::to_string(it) + ": " + std::to_string(h_val) + " ");
  }
} // sgld
} // namespace pift
#endif // PIFT_OPT_HPP
