// This is the Hamiltonian of Example 2 of the paper.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/21/2022
//


#ifndef PIFT_EXAMPLE02_HPP
#define PIFT_EXAMPLE02_HPP

#include <cmath>

#include "pift.hpp"

// This is a nonlinear diffusion Hamiltonian. The only parameter is
// the log of beta.
template<typename T>
class Example02Hamiltonian : public pift::Hamiltonian<T> {
  private:
    // The diffusion coefficient
    const T D;

    // The non-linear parameter
    const T kappa;

  public:
    Example02Hamiltonian(const T& D=0.1, const T& kappa=1.0) : 
      pift::Hamiltonian<T>(1), D(D), kappa(kappa)
    {}

    // The source term
    inline T f(const T* x) const
    {
      return std::cos(4.0 * x);
    }

    inline T operator()(const T* x, const T* prolong_phi, const T* theta) const 
    {
      const T beta = std::exp(theta[0]);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      return beta * (0.5 * D * phi_prime * phi_prime
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f(x));
    }
    
    inline T operator()(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
    ) const
    {
      const T beta = std::exp(theta[0]);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T f_x = f(x);
      const T tmp1 = kappa * pow(phi, 3)
      out[0] = beta * (tmp1 + f_x);
      out[1] = beta * D * phi_prime;
      return 0.5 * out[1] * phi_prime + (0.25 * tmp1 + f_x) * phi;
    }

    inline T add_grad_theta(
      const T* x, 
      const T* prolong_phi,
      const T* theta,
      T* out
    ) const {
      const T beta = std::exp(theta[0]);
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T result = beta * (0.5 * D * phi_prime * phi_prime
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f(x));
      out[0] += result;
      return result;
    }
}; // Example02Hamiltonian

#endif // PIFT_EXAMPLE02_HPP
