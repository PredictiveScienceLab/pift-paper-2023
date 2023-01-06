// Implementation of a fourier series.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022

#ifndef PIFT_FOURIER_HPP
#define PIFT_FOURIER_HPP

#include <cmath>
#include <vector>

#include "field.hpp"

namespace pift {

// A class representing a 1D field parameterized by a Fourier basis.
// Example
//
// Assume that domain is a UniformedRectangularDomain<float>
//
// using Domain = UniformedRectangularDomain<float>
// 
// phi =  Fourier1DField<float, Domain>(domain, num_terms);
//
template<typename T, typename D>
class Fourier1DField : public ParameterizedField<T> {
  protected:
    const int num_terms;
    D& domain;
    int dim_w;

  public:
    Fourier1DField(D& domain, const int& num_terms) :
      ParameterizedField<T>(1, 1 + (num_terms - 1) * 2, 1),
      domain(domain), num_terms(num_terms)
    {
      // TODO: Find a better way to get this info from the base class.
      dim_w = ParameterizedField<T>::dim_w;
    }

    inline T a() const { return domain.a(0); }
    inline T b() const { return domain.b(0); }
    inline T L() const { return domain.get_volume(); }

    // Evaluate the function approximation at x
    // Parameters:
    // x -- The point on which to evaluate
    // w -- The weight vector of size dim
    T operator()(const T* x, const T* w) const {
      T s = w[0];
      const T dx = 2.0 * M_PI / L() * (x[0] - a());
      for(int i=1; i<num_terms; i++) {
        const T tmp = dx * i;
        s += w[i] * std::cos(tmp) + w[num_terms + i - 1] * std::sin(tmp);
      }
      return s;
    }
  
    inline void operator()(const T* x, const T* w, T* prolong) const {
      prolong[0] = w[0];
      prolong[1] = 0.0;
      const T tmp1 =  2 * M_PI / L();
      const T dx = (x[0] - a());
      for(int i=1; i<num_terms; i++) {
        const T omega = tmp1 * i;
        const T omega_x = omega * dx;
        const T cos_omega_x = std::cos(omega_x);
        const T sin_omega_x = std::sin(omega_x);
        const T omega_cos_omega_x = omega * cos_omega_x;
        const T omega_sin_omega_x = omega * sin_omega_x;
        prolong[0] += w[i] * cos_omega_x + w[num_terms + i -1] * sin_omega_x;
        prolong[1] += (-w[i] * omega_sin_omega_x 
            + w[num_terms + i - 1] * omega_cos_omega_x);
      }
    }

    // Evaluate the gradient of f with respect to w
    inline T eval_grad(const T* x, const T* w, T* grad_w) const {
      T f = w[0];
      grad_w[0] = 1.0;
      const T tmp1 =  2 * M_PI / L();
      const T dx = (x[0] - a());
      for(int i=1; i<num_terms; i++) {
        const T omega = tmp1 * i;
        const T omega_x = omega * dx;
        const T cos_omega_x = std::cos(omega_x);
        const T sin_omega_x = std::sin(omega_x);
        const T omega_cos_omega_x = omega * cos_omega_x;
        const T omega_sin_omega_x = omega * sin_omega_x;
        f += w[i] * cos_omega_x + w[num_terms + i -1] * sin_omega_x;
        grad_w[i] = cos_omega_x;
        grad_w[num_terms + i - 1] = sin_omega_x;
      }
      return f;
    }

    // Evaluate the gradient of f with respect to w
    // and the gradient of f_prime with respect to w.
    inline void operator()(
      const T* x, const T* w, T* prolong, T* grad_w_prolong
    ) const {
      prolong[0] = w[0];
      prolong[1] = 0.0;
      grad_w_prolong[0] = 1.0;
      grad_w_prolong[dim_w] = 0.0;
      const T tmp1 =  2 * M_PI / L();
      const T dx = (x[0] - a());
      for(int i=1; i<num_terms; i++) {
        const T omega = tmp1 * i;
        const T omega_x = omega * dx;
        const T cos_omega_x = std::cos(omega_x);
        const T sin_omega_x = std::sin(omega_x);
        const T omega_cos_omega_x = omega * cos_omega_x;
        const T omega_sin_omega_x = omega * sin_omega_x;
        prolong[0] += w[i] * cos_omega_x + w[num_terms + i -1] * sin_omega_x;
        grad_w_prolong[i] = cos_omega_x;
        grad_w_prolong[num_terms + i - 1] = sin_omega_x;
        prolong[1] += (-w[i] * omega_sin_omega_x
            + w[num_terms + i - 1] * omega_cos_omega_x);
        grad_w_prolong[dim_w + i] = -omega_sin_omega_x;
        grad_w_prolong[dim_w + num_terms - 1 + i] = omega_cos_omega_x;
      }
    }
 
}; // Fourier1DField

} // namespace pift
#endif // PIFT_FOURIER_HPP
