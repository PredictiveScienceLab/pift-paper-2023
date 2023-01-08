// Radial basis function parameterization of fields.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  1/6/2023
//

#ifndef PIFT_RBF_HPP
#define PIFT_RBF_HPP

#include <cmath>
#include <vector>
#include <exception>

#include "field.hpp"

namespace pift {

// Template class representing a radial basis function parameterization
template<typename T, typename D>
class RBF1DField : public ParameterizedField<T,D> {
  protected:
    // The lengthscale
    const T ell;
    // The centers
    std::vector<T> centers;
    int dim_w;
    // Space to write
    std::vector<T> mu;

  public:
    RBF1DField(const std::vector<T>& centers, const T& ell) :
      ParameterizedField<T,D>(1, centers.size(), 1),
      centers(centers),
      ell(ell)
    {
      dim_w = ParameterizedField<T,D>::dim_w;
      mu.resize(dim_w);
    }

    inline T get_ell() const { return ell; }
    inline const std::vector<T>& get_centers() const { return centers; }

    T operator()(const T* x, const T* w) const {
      T r = T(0.0);
      for(int i=0; i<dim_w; i++)
        r += w[i] * std::exp(-0.5 * std::pow((x[0] - centers[i])/ell, 2));
      return r;
    }
  
    inline void operator()(const T* x, const T* w, T* prolong) const {
      throw std::runtime_error("Not implemented.");
    }

    inline T eval_grad(const T* x, const T* w, T* grad_w) const {
      T r = T(0.0);
      grad_w[0] = T(1.0);
      for(int i=0; i<dim_w; i++) {
        grad_w[i] = std::exp(-0.5 * std::pow((x[0] - centers[i])/ell, 2));
        r += w[i] * grad_w[i];
      }
      return r;
    }

    inline void operator()(
      const T* x, const T* w, T* prolong, T* grad_w_prolong
    ) const {
      throw std::runtime_error("Not implemented.");
    }

    inline void integrate_kernels(const T& a, const T& b, T* out) const {
      out[0] = b - a;
      for(int i=1; i<dim_w; i++)
        out[i] = ell * (std::erf((b - centers[i-1]) / std::sqrt(2.0) / ell)
                        -
                        std::erf((a - centers[i-1]) / std::sqrt(2.0) / ell)
            ) / std::sqrt(2.0 / M_PI);
    }

    inline T integrate(const D& domain, const T* w) const {
      T s = T(0.0);
      for(int i=0; i<dim_w; i++)
        s += w[i] * ell * (
            std::erf((domain.b(0) - centers[i]) / std::sqrt(2.0) / ell) -
            std::erf((domain.a(0) - centers[i]) / std::sqrt(2.0) / ell)
            ) / std::sqrt(2.0 / M_PI);
      return s;
    }

    inline T integrate(const D& domain, const T* w, T* grad_w) const {
      grad_w[0] = T(1.0);
      std::fill(grad_w + 1, grad_w + dim_w, T(0.0));
      return w[0];
      T s = T(0.0);
      for(int i=0; i<dim_w; i++) {
        grad_w[i] = ell * (
            std::erf((domain.a(0) - centers[i]) / std::sqrt(2.0) / ell) -
            std::erf((domain.b(0) - centers[i]) / std::sqrt(2.0) / ell)
            ) / std::sqrt(2.0 / M_PI);
        s += w[i] * grad_w[i];
      }
      return s;
    }
}; // RBF1DField 

} // namespace pift
#endif // PIFT_RBF_HPP
