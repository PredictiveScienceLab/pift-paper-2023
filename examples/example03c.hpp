// Example 3 of the paper.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  1/3/2023
//


#ifndef PIFT_EXAMPLE03C_HPP
#define PIFT_EXAMPLE03C_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "pift.hpp"

template<typename T, typename FA>
class Example03CHamiltonian : public pift::Hamiltonian<T> {
  private:
    const T f0;
    // The inverse temperature
    const T beta;
    // The precision of the source term weights
    const T tau;
    FA& f;
    std::vector<T> mu;
    std::vector<T> w;
    std::vector<T> grad_w;

    inline void update_local_w(const T* w_new) {
      T s = T(0.0);
      for(int i=1; i<f.get_dim_w(); i++)
        s += w_new[i - 1] * mu[i];
      w[0] = (f0 - s) / mu[0];
      std::copy(w_new, w_new + f.get_dim_w() - 1, w.begin() + 1);
    }

  public:
    Example03CHamiltonian(const T& beta, FA& f, const T& tau=10) : 
      pift::Hamiltonian<T>(2 + f.get_dim_w() - 1), beta(beta), f(f),
      tau(tau), 
      f0(0.25 * std::sin(4.0))
    {
      assert(f.get_dim_x() == 1);
      mu.resize(f.get_dim_w());
      w.resize(f.get_dim_w());
      grad_w.resize(f.get_dim_w());
      f.integrate_kernels(T(0.0), T(1.0), mu.data());
    }

    inline T get_tau(const T* theta) const { return tau; }
    inline T get_beta(const T* theta) const { return beta; }
    inline T get_D(const T* theta) const { return std::exp(theta[0]); }
    inline T get_kappa(const T* theta) const { return std::exp(theta[1]); }

    inline T operator()(const T* x, const T* prolong_phi, const T* theta) 
    {
      const T D = get_D(theta);
      const T kappa = get_kappa(theta);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      update_local_w(theta + 2);
      T pw = T(0.0);
      for(int i=0; i<f.get_dim_w() - 1; i++)
        pw += 0.5 * tau * w[i + 1] * w[i + 1];
      return beta * (0.5 * D * std::pow(phi_prime, 2)
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f(x, w.data()))
          + pw; 
    }
    
    inline T operator()(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
    ) {
      const T D = get_D(theta);
      const T kappa = get_kappa(theta);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      std::copy(theta + 2, theta + 1 + f.get_dim_w(), w.begin() + 1);
      const T f_x = f(x, w.data());
      const T tmp1 = beta * kappa * std::pow(phi, 3);
      const T tmp2 = beta * f_x;
      out[0] = tmp1 + tmp2;
      out[1] = beta * D * phi_prime;
      update_local_w(theta + 2);
      T pw = T(0.0);
      for(int i=0; i<f.get_dim_w() - 1; i++)
        pw += 0.5 * tau * w[i + 1] * w[i + 1];
      return 0.5 * out[1] * phi_prime + (0.25 * tmp1 + tmp2) * phi
        + pw;
    }

    inline T add_grad_theta(
      const T* x, 
      const T* prolong_phi,
      const T* theta,
      T* out
    ) {
      const T D = get_D(theta);
      const T kappa = get_kappa(theta);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      update_local_w(theta + 2);
      const T f_x = f.eval_grad(x, w.data(), grad_w.data());
      out[0] += 0.5 * beta * std::pow(phi_prime, 2) * D; 
      out[1] += 0.25 * beta * std::pow(phi, 4) * kappa;
      T pw = T(0.0);
      for(int i=0; i<f.get_dim_w() - 1; i++) {
        pw += 0.5 * tau * w[i + 1] * w[i + 1];
        out[2 + i] += beta * phi * grad_w[i + 1] + tau * w[i + 1];
      }
      return beta * (0.5 * D * std::pow(phi_prime, 2)
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f_x)
          + pw; 
    }
}; // Example03CHamiltonian
#endif // PIFT_EXAMPLE03C_HPP
