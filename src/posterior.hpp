// Template classes about posteriors.
//
// Author:
//  Ilias Bilionis
// 
// Date:
//  12/21/2022
//
//  TODO:
//    - Develop prior for the field weights.
//    - Develop prior for the physical parameters.
//

#ifndef PIFT_POSTERIOR_HPP
#define PIFT_POSTERIOR_HPP

#include <fstream>
#include <random>

#include "hamiltonian.hpp"
#include "likelihood.hpp"

namespace pift {

// An unbiased estimator for minus log posterior of w conditional on the
// data on theta.
template<typename T, typename UEH, typename UEL>
class UEGradWPost {
protected:
  UEH& prior;
  UEL& likelihood;
  const int dim_w;
  T* tmp;
  std::uniform_int_distribution<int>* unif_int;

public:
  UEGradWPost(UEH& prior, UEL& likelihood) : 
    prior(prior), likelihood(likelihood),
    dim_w(likelihood.get_dim_w())
  {
    tmp = new T[dim_w];
  }

  ~UEGradWPost() {
    delete tmp;
  }

  inline void set_theta(const T* theta) {
    prior.set_theta(theta);
    likelihood.set_theta(theta);
  }
  inline UEH& get_prior() { return prior; }
  inline UEL& get_likelihood() { return likelihood; }

  inline T operator()(const T* w, T* out) {
    const T p = prior(w, out);
    const T l = likelihood(w, tmp);
    for(int i=0; i<dim_w; i++)
      out[i] += tmp[i];
    return p + l;
  }
}; // UEGradWPost

// The parameters that control the behavior of UEGradWThetaHF.
template <typename T>
struct UEThetaParams {
  // The number of chains
  int num_chains;
  // The number of initial warmup steps
  int num_init_warmup;
  // The number of warmup steps per iteration
  int num_per_it_warmup;
  // The number of bursts (aka number of (almost) independent samples)
  int num_bursts;
  // The number of samples between bursts
  int num_thinning;
  // The variance for initializing sigma
  T init_w_sigma;
  // Whether or not to reinitialize the ws on every iteration
  bool reinitialize_ws;
  // The parameters used in SGLD
  SGLDParams<T> sgld_params; 

  UEThetaParams() :
    num_chains(1),
    num_init_warmup(10000),
    num_per_it_warmup(1),
    num_bursts(1),
    num_thinning(1),
    init_w_sigma(1.0),
    reinitialize_ws(false),
    sgld_params(SGLDParams<T>())
  {}
}; // UEThetaParams

// A class representing an unbiased estimator of the expectation of the
// gradient of a function. 
// The expectation is w with a distribution implied by a Hamiltonian.
// 
// Mathematically, it is:
// \[
//    \mathbb{E}\left[
//     \int_{\Omega}dx\nabla_{\theta}f(x;w),\theta)
//    \right] \approx
//      \frac{1}{N}\sum_{i=1}^N\nabla_{\theta}f(x;w_i),
// \]
// where the \(w_i\)'s are iid sampled from the distribution implied by
// `ue_grad_w_h` (the unbiased estimator of the gradient of a Hamiltonian).
//
template <typename T,
         typename UEGradWH,
         typename UEGradThetaF, typename R>
class UEGradThetaHF {
protected:
  UEGradWH& ue_grad_w_h;
  UEGradThetaF& ue_grad_theta_f;
  R& rng;
  const int dim_w;
  const int num_params;
  const T scale_ratio;
  UEThetaParams<T>& params;
  T* ws;
  T* grad_w_H;
  T* tmp_grad_theta;
  std::normal_distribution<T>* norm;
  std::ofstream of;

public:
  UEGradThetaHF(
      UEGradWH& ue_grad_w_h,
      UEGradThetaF& ue_grad_theta_f,
      R& rng,
      UEThetaParams<T>& params
  ) :
    ue_grad_w_h(ue_grad_w_h),
    ue_grad_theta_f(ue_grad_theta_f),
    rng(rng),
    params(params),
    dim_w(ue_grad_theta_f.get_dim_w()),
    num_params(ue_grad_theta_f.get_num_params()),
    scale_ratio(1.0 / (params.num_bursts * params.num_chains))      
  {
    grad_w_H = new T[dim_w];
    ws = new T[params.num_chains * dim_w];
    tmp_grad_theta = new T[num_params];
    norm = new std::normal_distribution<T>(0, 1);
    initialize_chains();
    std::cout << "save? " << params.sgld_params.save_to_file << std::endl;
    // REMOVE
    if (params.sgld_params.save_to_file)
      of.open(params.sgld_params.out_file);
  }

  ~UEGradThetaHF() {
    delete grad_w_H;
    delete ws;
    delete tmp_grad_theta;
    delete norm;
    if (params.sgld_params.save_to_file)
      of.close();
  }

  inline T get_scale_ratio() const { return scale_ratio; }
  inline int get_num_params() const { return num_params; }

  inline void initialize_chains() {
    for(int i=0; i<params.num_chains * dim_w; i++)
      ws[i] = params.init_w_sigma * (*norm)(rng);
  }

  inline void warmup(T* theta) {
    ue_grad_w_h.set_theta(theta);
    params.sgld_params.init_it = 0;
    for(int c=0; c<params.num_chains; c++) {
      std::cout << "*** doing chain: " << c << std::endl; // REMOVE
      T* w = ws + c * dim_w;
      sgld(
          ue_grad_w_h,
          w,
          dim_w,
          rng,
          params.num_init_warmup,
          grad_w_H,
          *norm,
          of,
          params.sgld_params
      );
      std::cout << "*** done" << std::endl;
    }
    params.sgld_params.init_it = params.num_init_warmup;
  }

  T operator()(T* theta, T* grad_theta) {
    T result = 0.0;
    std::fill(grad_theta, grad_theta + num_params, 0.0);
    if (params.reinitialize_ws)
      initialize_chains();
    ue_grad_w_h.set_theta(theta);
    const int init_it = params.sgld_params.init_it;
    for(int c=0; c<params.num_chains; c++) {
      T* w = ws + c * dim_w;
      params.sgld_params.init_it = init_it;
      sgld(
          ue_grad_w_h,
          w,
          dim_w,
          rng,
          params.num_per_it_warmup,
          grad_w_H,
          *norm,
          of,
          params.sgld_params
      );
      params.sgld_params.init_it += params.num_per_it_warmup;
      for(int b=0; b<params.num_bursts; b++) {
        // Sample w num_thinning times
        sgld(
            ue_grad_w_h,
            w,
            dim_w,
            rng,
            params.num_thinning,
            grad_w_H,
            *norm,
            of,
            params.sgld_params
        );
        params.sgld_params.init_it += params.num_thinning;
        // Get the gradient with respect to theta
        result += ue_grad_theta_f(w, theta, tmp_grad_theta);
        for(int i=0; i<num_params; i++)
          grad_theta[i] += tmp_grad_theta[i];
      } // end for over bursts
    } // end for over chains
    scale(grad_theta, num_params, scale_ratio);
    result *= scale_ratio;
    return result;
  }
}; 

// An unbiased estimator of minus the log posterior of theta conditional on
// the data
template <typename T, typename UEGradThetaPrior, typename UEGradThetaPost>
class UEGradThetaMinusLogPost {
public:
  UEGradThetaPrior& ue_prior;
  UEGradThetaPost& ue_post;
  T* grad_theta_prior; // Do I need this?
  int num_params; // Do I need this?
  UEGradThetaMinusLogPost(
      UEGradThetaPrior& ue_prior,
      UEGradThetaPost& ue_post
  ) : ue_prior(ue_prior), ue_post(ue_post),
      num_params(ue_prior.get_num_params())
  {
    grad_theta_prior = new T[num_params];
  }

  ~UEGradThetaMinusLogPost() {
    delete ue_prior;
    delete ue_post;
    delete grad_theta_prior;
  }

  inline int get_num_params() const { return num_params; }

  T operator()(const T* theta, T* grad_theta) {
    const T h_prior = ue_prior(theta, grad_theta_prior);
    const T h_post = ue_post(theta, grad_theta);
    for(int i=0; i<num_params; i++)
      grad_theta[i] -= grad_theta_prior[i];
    return h_post - h_prior;
  }
}; // UEGradThetaMinusLogPost

} // namespace pift
#endif // PIFT_POSTERIOR_HPP
