// This is the Hamiltonian of Example 2 of the paper when the error is in the
// energy.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/21/2022
//  1/3/2023
//


#ifndef PIFT_EXAMPLE02B_HPP
#define PIFT_EXAMPLE02B_HPP

#include <cmath>

#include "pift.hpp"

// This is a nonlinear diffusion Hamiltonian. The only parameter is
// the g(beta).
template<typename T>
class Example02BHamiltonian : public pift::Hamiltonian<T> {
  private:
    // The diffusion coefficient
    const T D;

    // The non-linear parameter
    const T kappa;
    
    // The parameter controlling how wrong the physics are
    const T gamma;

  public:
    Example02BHamiltonian(const T& gamma=1.0f, const T& D=0.1f, const T& kappa=1.0f) : 
      pift::Hamiltonian<T>(1), gamma(gamma), D(D), kappa(kappa)
    {}

    inline T get_beta(const T* theta) const {
      //return theta[0];
      return std::exp(theta[0]);
    }

    // The source term
    inline T f(const T* x) const
    {
      return std::cos(4.0f * x[0]);
    }

    inline T operator()(const T* x, const T* prolong_phi, const T* theta) 
    {
      const T beta = get_beta(theta);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      return beta * (0.5 * D * phi_prime * phi_prime
          + gamma * 0.25 * kappa * std::pow(phi, 4)
          + (1.0 - gamma) * 0.5 * std::pow(phi, 2)
          + phi * f(x));
    }
    
    inline T operator()(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
    )
    {
      const T beta = get_beta(theta);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T f_x = f(x);
      const T tmp1 = gamma * kappa * pow(phi, 3);
      const T tmp2 = (1.0 - gamma) * phi;
      out[0] = beta * (tmp1 + tmp2 + f_x);
      out[1] = beta * D * phi_prime;
      return 0.5 * out[1] * phi_prime + beta * (0.25 * tmp1 + 0.5 * tmp2 + f_x) * phi;
    }

    inline T add_grad_theta(
      const T* x, 
      const T* prolong_phi,
      const T* theta,
      T* out
    ) {
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T beta = get_beta(theta);
      const T res = beta * (0.5 * D * phi_prime * phi_prime
          + 0.25 * gamma * kappa * std::pow(phi, 4)
          + 0.5 * (1.0 - gamma) * std::pow(phi, 2)
          + phi * f(x));
      //const T res = 0.5 * D * phi_prime * phi_prime
      //    + 0.25 * kappa * std::pow(phi, 4)
      //    + phi * f(x);
      out[0] += res;
      return res;
    }
}; // Example02Hamiltonian

#endif // PIFT_EXAMPLE02_HPP
