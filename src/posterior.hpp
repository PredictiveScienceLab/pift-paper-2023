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
#include <vector>

#include "hamiltonian.hpp"
#include "likelihood.hpp"
#include "io.hpp"

namespace pift {

// An unbiased estimator for minus log posterior of w conditional on the
// data on theta.
template<typename T, typename UEH, typename UEL>
class UEGradWPost {
protected:
  UEH& prior;
  UEL& likelihood;
  const int dim_w;
  std::vector<T> tmp; 
  //std::uniform_int_distribution<int>* unif_int;

public:
  UEGradWPost(UEH& prior, UEL& likelihood) : 
    prior(prior), likelihood(likelihood),
    dim_w(likelihood.get_dim_w())
  {
    tmp.resize(dim_w);
  }

  ~UEGradWPost() {
  }

  inline void set_theta(T* theta) {
    prior.set_theta(theta);
    likelihood.set_theta(theta);
  }
  inline T get_beta(const T* theta) const {
    //return prior.get_beta(theta);
    return std::max(prior.get_beta(theta), likelihood.get_beta(theta));
  }
  inline UEH& get_prior() { return prior; }
  inline UEL& get_likelihood() { return likelihood; }

  inline T operator()(const T* w, T* out) {
    const T p = prior(w, out);
    const T l = likelihood(w, tmp.data());
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
  // Whether or not to adjust the SGLD alpha by dividing it by beta
  bool adjust_alpha;
  // Whether or not to reinitialize the ws on every iteration
  bool reinitialize_ws;
  // The prefix used for saving iles
  std::string output_prefix;
  // The parameters used in SGLD
  SGLDParams<T> sgld_params; 

  UEThetaParams() :
    num_chains(1),
    num_init_warmup(10000),
    num_per_it_warmup(1),
    num_bursts(1),
    num_thinning(1),
    init_w_sigma(1.0),
    adjust_alpha(true),
    reinitialize_ws(false),
    output_prefix("theta_default_output_prefix"),
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
  const T alpha0;
  UEThetaParams<T>& params;
  std::vector<T> ws;
  std::vector<T> grad_w_H;
  std::vector<T> tmp_grad_theta;
  std::normal_distribution<T> norm{0,1};
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
    alpha0(params.sgld_params.alpha),
    dim_w(ue_grad_theta_f.get_dim_w()),
    num_params(ue_grad_theta_f.get_num_params()),
    scale_ratio(1.0 / (params.num_bursts * params.num_chains))      
  {
    ws.resize(params.num_chains * dim_w);
    grad_w_H.resize(dim_w);
    tmp_grad_theta.resize(num_params);
    initialize_chains();
    if(params.sgld_params.save_to_file)
      of.open(params.sgld_params.out_file);
  }

  ~UEGradThetaHF() {
    if(params.sgld_params.save_to_file)
      of.close();
  }

  inline UEThetaParams<T>& get_params() { return params; }
  inline T get_scale_ratio() const { return scale_ratio; }
  inline int get_num_params() const { return num_params; }

  inline void initialize_chains() {
    for(int i=0; i<params.num_chains * dim_w; i++)
      ws[i] = params.init_w_sigma / std::sqrt(static_cast<T>(dim_w)) * norm(rng);
  }

  inline void adjust_alpha(const T* theta) {
    if(params.adjust_alpha) {
      const T beta = ue_grad_w_h.get_beta(theta);
      params.sgld_params.alpha = alpha0 / beta;
    }
  }

  inline void warmup(T* theta) {
    ue_grad_w_h.set_theta(theta);
    adjust_alpha(theta);
    params.sgld_params.init_it = 0;
    //if(params.sgld_params.save_to_file)
    //  of << "# INITIAL WARMUP" << std::endl;
    //if(params.sgld_params.disp)
    //  std::cout << "INITIAL WARMUP" << std::endl;
    //pift::cout_vec(theta, num_params, of, "# THETA: ");
    for(int c=0; c<params.num_chains; c++) {
      T* w = ws.data() + c * dim_w;
      //if(params.sgld_params.save_to_file)
      //  of << "# CHAIN: " << c << std::endl;
      //if(params.sgld_params.disp)
      //  std::cout << "CHAIN: " << c << std::endl;
      sgld(
          ue_grad_w_h,
          w,
          dim_w,
          rng,
          params.num_init_warmup,
          grad_w_H.data(),
          norm,
          of,
          params.sgld_params
      );
    }
    params.sgld_params.init_it = params.num_init_warmup;
  }

  T operator()(T* theta, T* grad_theta) {
    T foo[num_params];
    T result = T(0.0);
    T r2 = T(0.0);
    std::fill(grad_theta, grad_theta + num_params, T(0.0));
    std::fill(foo, foo + num_params, T(0.0));
    if(params.reinitialize_ws)
      initialize_chains();
    //ue_grad_w_h.set_theta(theta);
    adjust_alpha(theta);
    const int init_it = params.sgld_params.init_it;
    //if(params.sgld_params.save_to_file)
    //   pift::cout_vec(theta, num_params, of, "# THETA: ");
    for(int c=0; c<params.num_chains; c++) {
      T* w = ws.data() + c * dim_w;
      params.sgld_params.init_it = init_it;
      //if(params.sgld_params.save_to_file) {
      //  of << "# CHAIN: " << c << std::endl;
      //  of << "# WARMUP" << std::endl;
      //}
      //if(params.sgld_params.disp)
      //  std::cout << "CHAIN: " << c << std::endl;
      sgld(
          ue_grad_w_h,
          w,
          dim_w,
          rng,
          params.num_per_it_warmup,
          grad_w_H.data(),
          norm,
          of,
          params.sgld_params
      );
      params.sgld_params.init_it += params.num_per_it_warmup;
      for(int b=0; b<params.num_bursts; b++) {
        //if(params.sgld_params.save_to_file)
        //  of << "# BURST: " << b << std::endl;
        //if(params.sgld_params.disp)
        //  std::cout << "BURST: " << b << std::endl;
        sgld(
            ue_grad_w_h,
            w,
            dim_w,
            rng,
            params.num_thinning,
            grad_w_H.data(),
            norm,
            of,
            params.sgld_params
        );
        params.sgld_params.init_it += params.num_thinning;
        // Get the gradient with respect to theta
        const T r = ue_grad_theta_f(w, theta, tmp_grad_theta.data());
        if(params.sgld_params.save_to_file)
          pift::cout_vec(
              tmp_grad_theta,
              of,
              "# SAMPLE_PRIOR_EXP_INT_GRAD_THETA_H: "
          );
        result += r;
        r2 += r * r;
        for(int i=0; i<num_params; i++) {
          foo[i] += std::pow(tmp_grad_theta[i], 2);
          grad_theta[i] += tmp_grad_theta[i];
        }
      } // end for over bursts
    } // end for over chains
    params.sgld_params.init_it = init_it + params.num_per_it_warmup 
      + params.num_bursts * params.num_thinning;
    scale(grad_theta, num_params, scale_ratio);
    if(params.sgld_params.save_to_file)
           pift::cout_vec(
              grad_theta,
              num_params,
              of,
              "# UE_PRIOR_EXP_INT_GRAD_THETA_H: "
           );
    result *= scale_ratio;
    scale(foo, num_params, scale_ratio);
    for(int i=0; i<num_params; i++)
      foo[i] -= std::pow(grad_theta[i], 2);
    return result;
  }
}; 

// An unbiased estimator of minus the log posterior of theta conditional on
// the data
template <typename T, typename UEGradThetaPrior, typename UEGradThetaPost>
class UEGradThetaMinusLogPost {
private:
  UEGradThetaPrior& ue_prior;
  UEGradThetaPost& ue_post;
  std::vector<T> grad_theta_prior;
  int num_params; // Do I need this?

public:
  UEGradThetaMinusLogPost(
      UEGradThetaPrior& ue_prior,
      UEGradThetaPost& ue_post
  ) : ue_prior(ue_prior), ue_post(ue_post),
      num_params(ue_prior.get_num_params())
  {
    grad_theta_prior.resize(num_params);
  }

  inline UEGradThetaPrior& get_prior() { return ue_prior; }
  inline UEGradThetaPost& get_post() { return ue_post; }
  inline int get_num_params() const { return num_params; }
  inline void initialize_chains() {
    ue_prior.initialize_chains();
    ue_post.initialize_chains();
  }
  inline void warmup(T* theta) {
    ue_prior.warmup(theta);
    ue_post.warmup(theta);
  }

  T operator()(T* theta, T* grad_theta) {
    grad_theta_prior[0] = -1.0;
    const T h_prior = ue_prior(theta, grad_theta_prior.data());
    const T h_post = ue_post(theta, grad_theta);
    for(int i=0; i<num_params; i++)
      grad_theta[i] -= grad_theta_prior[i];
    return h_post - h_prior;
  }
}; // UEGradThetaMinusLogPost


// The following functionality requires restructoring the code
// TODO: Fix me.
template<
  typename T,
  typename H,
  typename FA,
  typename D,
  typename RNG,
  typename C>
auto make_ue_prior_exp_int_grad_theta(
    H& h,
    FA& phi,
    D& domain,
    RNG& rng,
    C& config,
    T* theta,
    std::string& prefix
) {
  // The unbiased estimator of the integral of the gradient of the
  // Hamiltonian with respect to theta
  using UEGradThetaH = UEIntegralGradThetaH<T, H, FA, D>;
  UEGradThetaH ue_int_grad_theta_H(
      h,
      phi,
      domain,
      config.num_collocation
  );

  // Unbiased estimator used to take expectations over prior
  using UEGradWH = UEIntegralGradWH<T, H, FA, D>;
  UEGradWH ue_grad_w_h(
      h,
      phi,
      domain,
      config.num_collocation,
      theta
  );

  // Unbiased estimator of the prior expectation of the integral of grad theta
  // of the Hamiltonian
  auto theta_params = config.get_theta_params();
  theta_params.sgld_params.out_file = prefix + "_prior_ws.csv";
  using UEGradThetaPrior = UEGradThetaHF<T, UEGradWH, UEGradThetaH, RNG>;
  UEGradThetaPrior ue_prior_exp_int_grad_theta_H(
      ue_grad_w_h,
      ue_int_grad_theta_H,
      rng,
      theta_params
  );
  //UEGradThetaPrior* ue_prior_exp_int_grad_theta_H = new UEGradThetaPrior(
  //    ue_grad_w_h,
  //    ue_int_grad_theta_H,
  //    rng,
  //    theta_params
  //);

  return ue_prior_exp_int_grad_theta_H;
} // make_ue_prior_exp_int_grad_theta
} // namespace pift
#endif // PIFT_POSTERIOR_HPP
