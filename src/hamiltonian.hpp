// Template classes related to Hamiltonians.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/16/2022
//
// TODO:
//    - This design of the Hamiltonian does redundant computations when the
//      parameters stay fixed. A better design could have been to allow the
//      user to set_theta which would trigger some computations.
//

#ifndef PIFT_HAMILTONIAN_HPP
#define PIFT_HAMILTONIAN_HPP

namespace pift {

#include <vector>
#include <algorithm>

// An abstract class representing a Hamiltonian. It does not really do
// anything. It just defines the interface that any actual Hamiltonian must
// adhere to.
template<typename T>
class Hamiltonian {
protected:
  const int num_params;

public:
  Hamiltonian(const int& num_params) : num_params(num_params)
  {}

  // All objects must have copy constructors like this one
  Hamiltonian(const Hamiltonian<T>& obj) : Hamiltonian<T>(obj.num_params) {}

  inline int get_num_params() const { return num_params; }

  // Return the beta (inverse temperature of the Hamiltonian) from knowledge
  // of the parameters.
  virtual inline T get_beta(const T* theta) const = 0;

  // The density of the Hamiltonian
  // This must be overloaded by the deriving class.
  // Parameters:
  //  x            --  The spatial point on which to evaluate the Hamiltonian
  //   prolong_phi  --  A vector representing the prolongation of the phi.
  //                   In the simplest case, 1D, this is (phi, phi_prime).
  //  theta        --  The physical parameters.
  // Returns:
  //  The value of the hamiltonian density
  virtual inline T operator()(
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
  virtual inline T operator()(
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
}; // Hamiltonian

// An unbiased estimator of the spatial integral of grad w of the Hamiltonian
// density.
//
// The mathematical formula is:
// \[
//  \int_{\Omega} dx \nabla_w h(x, \text{Pr}[\phi](x;w),\dots,\theta) \approx
//    \frac{1}{N}\sum_{i=1}^N 
//      \nabla_w h(x_i, \text{Pr}[\phi](x_i;w),\theta),
// \]
// where \(x_i\) are uniformly and independently sampled from \(\Omega\).
// 
// Template arguments:
//  T  -- The type of floating point numbers.
//  H  -- A Hamiltonian.
//  FA -- A function approximation
//  D  -- A spatial/time domain sampler
template<typename T, typename H, typename FA, typename D>
class UEIntegralGradWH {
  protected:
    H& h;
    FA& phi;
    D& domain;
    T* theta;
    const int num_collocation;
    const int dim_x;
    const int dim_w;
    const int prolong_size;
    const T scale_ratio;
    T* prolong_phi;
    T* grad_w_prolong_phi;
    T* grad_prolong_phi_H;
    T* x;

  public:
    // Initialize the object
    // Parameters:
    //  h       -- A Hamiltonian
    //  fa      -- A function approximation.
    //  domain  -- A spatial/time domain.
    //  num_collocation -- The number of collocation points to sample.
    //  theta   -- The parameters to keep fixed.
    UEIntegralGradWH(
        H& h, FA& phi, D& domain, const int& num_collocation, T* theta
    ) :
      h(h),
      phi(phi),
      domain(domain),
      dim_x(phi.get_dim_x()),
      dim_w(phi.get_dim_w()),
      prolong_size(phi.get_prolong_size()),
      num_collocation(num_collocation),
      scale_ratio(domain.get_volume() / num_collocation),
      theta(theta)
    {
      prolong_phi = new T[prolong_size];
      grad_w_prolong_phi = new T[phi.get_grad_w_prolong_size()];
      grad_prolong_phi_H = new T[prolong_size];
      std::fill(
          grad_prolong_phi_H,
          grad_prolong_phi_H + prolong_size,
          0.0
      );
      x = new T[dim_x];
    }

    ~UEIntegralGradWH() {
      delete prolong_phi;
      delete grad_w_prolong_phi;
      delete grad_prolong_phi_H;
      delete x;
    }

    // TODO: Think if we need to keep this
    inline void set_theta(T* theta) { this->theta = theta; }
    inline T get_beta(const T* theta) const { return h.get_beta(theta); }
    inline FA& get_phi() const { return phi; }
    inline int get_num_collocation() const { return num_collocation; }

    // Adds the gradient of the Hamiltonian density with respect to w to out
    // Returns the Hamiltonian density at x.
    inline T add_grad(const T* x, const T* w, T* out) {
      phi(x, w, prolong_phi, grad_w_prolong_phi);
      const T h_x = h(x, prolong_phi, theta, grad_prolong_phi_H);
      for(int i=0; i<dim_w; i++)
        for(int j=0; j<prolong_size; j++)
         out[i] += grad_prolong_phi_H[j] * grad_w_prolong_phi[dim_w * j + i]; 
      return h_x;
    }

    // Implements an unbiased estimator of the Hamiltonian
    // Returns the estimator.
    // In out, it writes the gradient of the estimator with respect to w
    inline T operator()(const T* w, T* out) {
      T s = 0.0;
      std::fill(out, out + dim_w, 0.0);
      for(int i=0; i<num_collocation; i++) {
        domain.sample(x);
        s += add_grad(x, w, out);
      }
      scale(out, dim_w, scale_ratio, out);
      return s * scale_ratio;
    }
}; // UEIntegralGradWH

// An unbiased estimator of the spatial integral of grad theta of the
// Hamiltonian.
//
// The formula is:
// \[
// \int_{\Omega}dx \nabla_{\theta} h(x,\text{Pr}[\phi](x,w),\theta) \approx
//  \frac{1}{N}\sum_{i=1}^N \nabla_{\theta}(x_i,\text{Pr}[\phi](x_i,w),\theta)
// \]
//
// TODO: Write unittest.
template<typename T, typename H, typename FA, typename D>
class UEIntegralGradThetaH {
  protected:
    H& h;
    FA& phi;
    D& domain;
    const int num_collocation;
    const int dim_x;
    const int dim_w;
    const int prolong_size;
    const T scale_ratio;
    T* prolong_phi;
    T* x;

  public:
    // Initialize the object
    // Parameters:
    //  h       -- A Hamiltonian
    //  fa      -- A function approximation.
    //  domain  -- A spatial/time domain.
    //  num_collocation -- The number of collocation points to sample.
    UEIntegralGradThetaH(
        H& h, FA& phi, D& domain, const int& num_collocation
    ) :
      h(h),
      phi(phi),
      domain(domain),
      dim_x(phi.get_dim_x()),
      dim_w(phi.get_dim_w()),
      prolong_size(phi.get_prolong_size()),
      num_collocation(num_collocation),
      scale_ratio(domain.get_volume() / num_collocation)
    {
      prolong_phi = new T[prolong_size];
      x = new T[dim_x];
    }

    ~UEIntegralGradThetaH() {
      delete prolong_phi;
      delete x;
    }

    inline FA& get_phi() const { return phi; }
    inline int get_num_collocation() const { return num_collocation; }
    inline int get_dim_w() const { return dim_w; }
    inline int get_dim_x() const { return dim_x; }
    inline int get_num_params() const { return h.get_num_params(); }

    // Adds the gradient of the Hamiltonian density with respect to w to out
    // Returns the Hamiltonian density at x.
    inline T add_grad(const T* x, const T* w, const T* theta, T* out) {
      phi(x, w, prolong_phi);
      return h.add_grad_theta(x, prolong_phi, theta, out);
    }

    // Returns the estimator.
    // In out, it writes the gradient of the estimator with respect to w
    inline T operator()(const T* w, const T* theta, T* out) {
      T s = 0.0;
      std::fill(out, out + dim_w, 0.0);
      for(int i=0; i<num_collocation; i++) {
        domain.sample(x);
        s += add_grad(x, w, theta, out);
      }
      scale(out, dim_w, scale_ratio, out);
      return s * scale_ratio;
    }
}; // UEIntegralGradThetaH
} // namespace pift
#endif // PIFT_HAMILTONIAN_HPP
