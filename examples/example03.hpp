// Example 3 of the paper.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  1/3/2023
//


#ifndef PIFT_EXAMPLE03_HPP
#define PIFT_EXAMPLE03_HPP

#include <cmath>
#include <vector>

#include "pift.hpp"

template<typename T, typename FA>
class Example03Hamiltonian : public pift::Hamiltonian<T> {
  private:
    // The inverse temperature
    const T beta;
    FA& f;
    std::vector<T> grad_w;

  public:
    Example03Hamiltonian(const T& beta, FA& f) : 
      pift::Hamiltonian<T>(2 + f.get_dim_w()), beta(beta)
    {
      assert(f.get_dim_x() == 1);
      grad_w.resize(f.get_dim_w());
    }

    inline T get_beta(const T* theta) const { return beta; }

    inline T operator()(const T* x, const T* prolong_phi, const T* theta) const 
    {
      const T D = theta[0];
      const T kappa = theta[1];
      const T* w = theta + 2;
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      return beta * (0.5 * D * std::pow(phi_prime, 2)
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f(x, w));
    }
    
    inline T operator()(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
    ) const
    {
      const T D = theta[0];
      const T kappa = theta[1];
      const T* w = theta + 2;
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T f_x = f(x, w);
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
    ) const {
      const T D = theta[0];
      const T kappa = theta[1];
      const T* w = theta + 2;
      const T phi = prolong_phi[0];
      const T phi_prime = prolong_phi[1];
      const T* f_x = f.eval_grad(x, w, grad_w.data());
      out[0] += 0.5 * beta * std::pow(phi_prime, 2);
      out[1] += 0.25 * beta * std::pow(phi, 4);
      for(int i=0; i<f.get_dim_w(); i++)
        out[2 + i] += beta * phi * grad_w[i];
      return beta * (0.5 * D * std::pow(phi_prime, 2)
          + 0.25 * kappa * std::pow(phi, 4)
          + phi * f_x);
      return res;
    }

}; // Example03Hamiltonian
#endif // PIFT_EXAMPLE03_HPP
