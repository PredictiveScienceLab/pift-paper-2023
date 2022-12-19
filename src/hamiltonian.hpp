/* Template classes related to Hamiltonians.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/16/2022
 *
 */

#ifndef PIFT_HAMILTONIAN_HPP
#define PIFT_HAMILTONIAN_HPP

#include <vector>

using namespace std;

template<typename T>
class Hamiltonian {
private:
  const int num_params;

public:
  Hamiltonian(const int& num_params) : num_params(num_params),
  {}

  inline int get_num_params() const { return num_params; }

  // The density of the Hamiltonian
  // This must be overloaded by the deriving class.
  // Parameters:
  //  x            --  The spatial point on which to evaluate the Hamiltonian
  //   prolong_phi  --  A vector representing the prolongation of the phi.
  //                   In the simplest case, 1D, this is (phi, phi_prime).
  //  theta        --  The physical parameters.
  // Returns:
  //  The value of the hamiltonian density
  virtual inline T density(
      const T* x, const T* prolong_phi, const T* theta
  ) const = 0;

  // The gradient of the Hamiltonian with respect to the prolongation of phi
  //  x             --  The spatial point on which to evaluate the Hamiltonian
  //   prolong_phi  --  A vector representing the prolongation of the phi.
  //                   In the simplest case, 1D, this is (phi, phi_prime).
  //  theta         --  The physical parameters.
  // Returns:
  //  The valua of the hamiltonian density
  //  out -- This is where we write the gradient of the Hamiltonian with
  //         respect to prolong_phi.
  virtual inline T grad_phi_prolong(
      const T* x, const T* prolong_phi, const T* theta,
      T* out
  ) const = 0;

  // Add the gradient of the Hamiltonian with respect to the phyisical
  // parameters theta
  virtual inline T add_grad_theta(
      const T* x, 
      const T* prolong_phi,
      const T* theta,
      T* out
  ) const = 0;
};

// A class representing an unbiased estimator of the gradient of
// Hamiltonian with resepect to w at a fixed theta.
// 
// Template arguments:
//  T  -- The type of floating point numbers.
//  H  -- A Hamiltonian.
//  FA -- A function approximation
//  D  -- A spatial/time domain sampler
template <typename T, typename H, typename FA, typename D>
class UEGradHAtFixedTheta {
  private:
    H& h;
    FA& fa;
    D& domain;
    const T* theta;
    const int num_collocation;
    T* prolong_phi;

  public:

    // Initialize the object
    // Parameters:
    //  h       -- A Hamiltonian
    //  fa      -- A function approximation.
    //  domain  -- A spatial/time domain.
    //  num_collocation -- The number of collocation points to sample.
    //  theta   -- The parameters to keep fixed.
    UEGradHAtFixedTheta(
        H& h, FA& fa, D& domain, const int& num_collocation, const T* theta
    ) :
      h(h),
      fa(fa),
      domain(domain),
      num_collocation(num_collocation),
      theta(theta)
    {
      prolong_phi = new T[h.num_params];
    }

    ~UEGradHAtFixedTheta() {
      delete prolong_phi;
    }

    inline int get_num_collocation() const { return num_collocation; }
  T add_grad_w(
    const T& x, const T* w,
    const T* theta,
    T* grad_w_H
  ) {
    T phi_val, phi_prime_val;
    phi.eval_grads(x, w, phi_val, grad_phi, phi_prime_val, grad_phi_prime);
    const T h = density(x, phi_val, phi_prime_val, theta);
    const T dh_dphi = dphi(x, phi_val, phi_prime_val, theta);
    const T dh_dphi_prime = dphi_prime(x, phi_val, phi_prime_val, theta);
    for(int i=0; i<dim; i++)
      grad_w_H[i] += dh_dphi * grad_phi[i] + dh_dphi_prime * grad_phi_prime[i];
    return h;
  }

    inline T add_grad_w(const T* x, const T* w, T* out) {
      fa(x, w, prolong_phi);
    }

    inline T operator()(const T* w, T* out) {
      
    }
};

#endif // PIFT_HAMILTONIAN_HPP
