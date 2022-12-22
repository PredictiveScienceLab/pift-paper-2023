// Diffusion Hamiltonian. This is example 1 of the paper.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022
//


#ifndef PIFT_EXAMPLE01_HPP
#define PIFT_EXAMPLE01_HPP

#include <cmath>

#include "pift.hpp"

template<typename T>
class Example01Hamiltonian : public pift::Hamiltonian<T> {
  private:
    // The inverse temperature
    const T beta;

    // The diffusion coefficient
    const T kappa;

  public:
    Example01Hamiltonian(const T& beta, const T& kappa=0.25) : 
      pift::Hamiltonian<T>(0), beta(beta), kappa(kappa)
  {}

    // The source term
    inline T q(const T* x) const
    {
      return std::exp(-x[0]);
    }

    inline T operator()(const T* x, const T* prolong_phi, const T* theta) const 
    {
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      return beta * (0.5 * kappa * phi_prime * phi_prime - phi * q(x));
    }
    
    inline T operator()(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
    ) const
    {
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T q_x = q(x);
      out[0] = -beta * q_x;
      out[1] = beta * kappa * phi_prime;
      return 0.5 * out[1] * phi_prime + out[0] * phi;
    }

    inline T add_grad_theta(
      const T* x, 
      const T* prolong_phi,
      const T* theta,
      T* out
    ) const {
      return this->operator()(x, prolong_phi, theta);
    }
}; // Example01Hamiltonian

#endif // PIFT_EXAMPLE01_HPP
