// Classes that represent a likelihood function.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/21/2022
//
// TODO:
//  - Develop the concept of measurement operator.
//  - Design abstract Likelihood class.
//  - Allow likelihood to have parameters.
//

#ifndef PIFT_LIKELIHOOD_HPP
#define PIFT_LIKELIHOOD_HPP

#include <random>
#include <algorithm>
#include <cmath>

#include "utils.hpp"

namespace pift {

// The following class works only for pointwise observations of a scalar field
// Also it cannot have parameters of its own.
//
template<typename T, typename FA>
class GaussianLikelihood {
protected:
  const int num_params;
  const FA& phi;
  const int dim_x;
  const int dim_w;
  const T sigma;
  const T sigma2;
  const T* x_obs;
  const T* y_obs;
  const int num_obs;
  T* grad_phi;

public:
  GaussianLikelihood(
    const FA& phi,
    const int& num_obs,
    const T* x_obs,
    const T* y_obs,
    const T& sigma
  ) :
    num_params(0),
    phi(phi),
    dim_x(phi.get_dim_x()),
    dim_w(phi.get_dim_w()),
    sigma(sigma),
    sigma2(std::pow(sigma, 2)),
    num_obs(num_obs),
    x_obs(x_obs),
    y_obs(y_obs)
  {
    grad_phi = new T[dim_w];
  }

  ~GaussianLikelihood() {
    delete grad_phi;
  }

  inline T get_beta(const T* theta) const { return 1.0 / sigma2; }
  inline int get_num_params() const { return num_params; }
  inline const FA& get_phi() const { return phi; }
  inline int get_dim_w() const { return dim_w; }
  inline T get_sigma() const { return sigma; }
  inline int get_num_obs() const { return num_obs; }
  inline const T* get_x_obs() const { return x_obs; }
  inline const T* get_y_obs() const { return y_obs; }

  // The minus log likelihood of one observation
  inline T operator()(const int& n, const T* w, const T* theta) const {
    return 0.5 * std::pow((phi->eval(x_obs[n], w) - y_obs[n]) / sigma, 2);
  }

  // The minus log likelihood of all observations
  inline T operator()(const T* w, const T* theta) const {
    T l = 0.0;
    for(int n=0; n<num_obs; n++)
      l += operator()(n, w);
    return l;
  }

  // Computes the minus of the log likelihood and the gradient of it with
  // respect to w. The latter is added to out.
  inline T add_grad_w(
    const int& n,
    const T* w,
    const T* theta,
    T* out
  ) {
    const T phi_n = phi.eval_grad(x_obs + n * dim_x, w, grad_phi);
    const T std_err = (phi_n - y_obs[n]) / sigma;
    const T std_err_over_sigma = std_err / sigma;
    // grad_w_minus_log_like = d_minus_log_like_d_phi * grad_w_phi
    for(int i=0; i<dim_w; i++)
      out[i] += std_err_over_sigma * grad_phi[i];
    return 0.5 * std::pow(std_err, 2);
  }

  // Computes the minus of the log likelihood of all observations
  inline T add_grad_w(const T* w, const T* theta, T* out) {
    T l = 0.0;
    for(int n=0; n<num_obs; n++)
      l += add_grad_w(n, w, theta, out);
    return l;
  }

}; // GaussianLikelihood

// This unbiased estimator only works with a Gaussian likelihood
// TODO: Generalize
template<typename T, typename L, typename R>
class UEGradWL {
  protected:
    L& l;
    T* theta;
    const int batch_size;
    const T scale_ratio;
    const int dim_w;
    R rng;
    std::uniform_int_distribution<int> unif_int;

  public:
    UEGradWL(
        L& l, T* theta, const int& batch_size, R& rng
    ) :
      l(l), theta(theta), batch_size(batch_size),
      rng(rng),
      dim_w(l.get_dim_w()),
      scale_ratio(static_cast<T>(l.get_num_obs()) / static_cast<T>(batch_size))
    {
      unif_int.param(
          std::uniform_int_distribution<int>::param_type(
            0, l.get_num_obs() - 1
          )
      );
    }

    inline T get_beta(const T* theta) const { return l.get_beta(theta) * l.get_num_obs(); }
    inline void set_theta(T* theta) { this->theta = theta; }
    inline int get_dim_w() const { return dim_w; }

    inline T operator()(const T* w, T* out) {
      std::fill(out, out + dim_w, 0.0);
      T s = 0.0;
      for(int m=0; m<batch_size; m++) {
        const int n = unif_int(rng);
        s += l.add_grad_w(n, w, theta, out);
      }
      s *= scale_ratio;
      pift::scale(out, dim_w, scale_ratio);
      return s;
    }

    inline R& get_rng() const { return rng; }
}; // UEGradWL
} // namespace pift
#endif // PIFT_LIKELIHOOD_HPP
