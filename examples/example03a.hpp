// Example 3.a of the paper.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  1/3/2023
//


#ifndef PIFT_EXAMPLE03A_HPP
#define PIFT_EXAMPLE03A_HPP

#include <cmath>
#include <vector>
#include <numeric>

#include "pift.hpp"

template<typename T>
class Example03AHamiltonian : public pift::Hamiltonian<T> {
  private:
    // The inverse temperature
    const T beta;

  public:
    Example03AHamiltonian(const T& beta): 
      pift::Hamiltonian<T>(2), beta(beta)
    {}

    inline T get_beta(const T* theta) const { return beta; }
    inline T get_D(const T* theta) const { return std::exp(theta[0]); }
    inline T get_kappa(const T* theta) const { return std::exp(theta[1]); }
    inline T f(const T* x) const { return std::cos(T(4.0) * x[0]); }

    inline T operator()(const T* x, const T* prolong_phi, const T* theta) 
    {
      const T D = get_D(theta);
      const T kappa = get_kappa(theta);
      const T* w = theta + 2;
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      return beta * (0.5 * D * std::pow(phi_prime, 2)
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f(x));
    }

    inline T operator()(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
    )
    {
      const T D = get_D(theta);
      const T kappa = get_kappa(theta);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T f_x = f(x);
      const T tmp1 = beta * kappa * std::pow(phi, 3);
      const T tmp2 = beta * f_x;
      out[0] = tmp1 + tmp2;
      out[1] = beta * D * phi_prime;
      return 0.5 * out[1] * phi_prime + (0.25 * tmp1 + tmp2) * phi;
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
      const T f_x = f(x);
      out[0] += 0.5 * beta * std::pow(phi_prime, 2) * D; 
      out[1] += 0.25 * beta * std::pow(phi, 4) * kappa;
      return beta * (0.5 * D * std::pow(phi_prime, 2)
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f_x);
    }

}; // Example03AHamiltonian
#endif // PIFT_EXAMPLE03A_HPP
