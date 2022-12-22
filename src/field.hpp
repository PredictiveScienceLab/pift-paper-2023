// Some generic template classes for fields.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022
//
//  TODO:
//    - Develop abstractions to deal with multi-dimensional fields.
//    - Develop template class for tensor product of fields.
//    - Develop template class for constraining 2D and 3D things at an 
//      arbitrary boundary.
//    - Develop fields with parameterized boundary.

#ifndef PIFT_FIELD_HPP
#define PIFT_FIELD_HPP

#include <algorithm>
#include <vector>
#include <cmath>
#include <cassert>

namespace pift {

// A class representing a parameterized field.
// This is an abstract class.
template<typename T>
class ParameterizedField {
  protected:
    const int dim_x;
    const int dim_w;
    const int max_deriv;
    int prolong_size;
    int grad_w_prolong_size;

  public:
    ParameterizedField(
        const int& dim_x, const int dim_w, 
        const int& max_deriv,
        const int& prolong_size=-1
    ) :
      dim_x(dim_x),
      dim_w(dim_w),
      max_deriv(max_deriv)
    {
      if(prolong_size == -1) {
        if (dim_x == 1)
          this->prolong_size = dim_x * (1 + max_deriv);
        else {
          this->prolong_size = (1 - std::pow(dim_x, max_deriv)) / (1 - dim_x);
        }
      } else
        this->prolong_size = prolong_size;
      grad_w_prolong_size = dim_w * this->prolong_size;
    }

    inline int get_dim_x() const { return dim_x; }
    inline int get_dim_w() const { return dim_w; }
    inline int get_max_deriv() const { return max_deriv; }
    inline int get_prolong_size() const { return prolong_size; }
    inline int get_grad_w_prolong_size() const { return grad_w_prolong_size; }

    // Evaluates the field at x and w
    virtual T operator()(const T* x, const T* w) const = 0;
    // Evaluates the prolongation of the field at x and w
    virtual void operator()(const T* x, const T* w, T* prolong) const = 0;
    // Evaluates the prolongation of the field at x and w along with 
    // the gradient of the prolongation with respect to w
    virtual void operator()(
        const T* x, const T* w, T* prolong, T* grad_w_prolong
    ) const = 0;
    // Evaluates the field and its gradient with respect to w
    virtual T eval_grad(const T* x, const T* w, T* grad_w) const = 0;
}; // ParameterizedField


// A class representing a parameterized 1D field that is constrained at the
// boundary.
//
// Example:
//
// Suppose phi is a ParameterizedField<float> with dim_x == 1.
// Also suppose that domain is a UniformRectangularDomain<float> with 1
// dimension.
//
//// Some definitions to simplify the templates:
//  using Domain = UniformedRectangularDomain<float>
//  using Field = ParameterizedField<float>
//  using CField = Constrained1DField<float, Field, Domain>;
//
//// The values on the boundary
/// float boundary_values[2] = {0.0, 1.0};
//  psi = CField(phi, domain, boundary_values);
//
template<typename T, typename PF, typename D>
class Constrained1DField : public ParameterizedField<T> {
   protected:
     const PF& phi;
     const D& domain;
     std::vector<T> values;
     int dim_w;
     int dim_x;
 
   public:
     Constrained1DField(
         const PF& phi, 
         const D& domain,
         const T* values
       ) :
       ParameterizedField<T>(
           phi.get_dim_x(), phi.get_dim_w(),
           phi.get_max_deriv(), phi.get_prolong_size()
       ),
       phi(phi),
       domain(domain)
     {
       // TODO: Find a better way to do this
       dim_x = ParameterizedField<T>::dim_x;
       dim_w = ParameterizedField<T>::dim_w;
       assert(dim_x == 1);
       this->values.assign(values, values + 2);
     }

     Constrained1DField(
         const PF& phi,
         const D& domain,
         const std::vector<T> values
     ) : Constrained1DField<T, PF, D>(phi, domain, values.data())
     {
       assert(values.size() == 2);
     }
 
     inline T a() const { return domain.a(0); }
     inline T ya() const { return values[0]; }
     inline T b() const { return domain.b(0); }
     inline T yb() const { return values[1]; }
 
     inline T operator()(const T* x, const T* w) const {
       const T xma = x[0] - a();
       const T bmx = b() - x[0];
       return bmx * ya() + xma * yb() + xma * bmx * phi(x, w);
     }
 
     inline void operator()(const T* x, const T* w, T* prolong) const {
       const T xma = x[0] - a();
       const T bmx = b() - x[0];
       const T xmabmx = xma * bmx;
       const T bmxmxma = bmx - xma;
       phi(x, w, prolong);
       prolong[1] = yb() - ya() + bmxmxma * prolong[0] + xmabmx * prolong[1];
       prolong[0] = bmx * ya() + xma * yb() + xmabmx * prolong[0];
     }  
 
     inline void operator()(
         const T* x, const T* w, T* prolong, T* grad_w_prolong
     ) const {
       const T xma = x[0] - a();
       const T bmx = b() - x[0];
       const T xmabmx = xma * bmx;
       const T bmxmxma = bmx - xma;
       phi(x, w, prolong, grad_w_prolong);
       prolong[1] = yb() - ya() + bmxmxma * prolong[0] + xmabmx * prolong[1];
       prolong[0] = bmx * ya() + xma * yb() + xmabmx * prolong[0];
       std::transform(grad_w_prolong + dim_w, grad_w_prolong + 2 * dim_w,
                 grad_w_prolong,
                 grad_w_prolong + dim_w,
                 [&bmxmxma, &xmabmx](T v1, T v2) {
                   return bmxmxma * v2 + xmabmx * v1;
                  });
       std::transform(grad_w_prolong, grad_w_prolong + dim_w,
                 grad_w_prolong,
                 [&xmabmx](T v) {return xmabmx * v;});
     }
 
     inline T eval_grad(const T* x, const T* w, T* grad_w) const {
      const T xma = x[0] - a();
      const T bmx = b() - x[0];
      const T xmabmx = xma * bmx;
      const T bmxmxma = bmx - xma;
      T f = phi.eval_grad(x, w, grad_w);
      f = bmx * ya() + xma * yb() + xmabmx * f;
      std::transform(grad_w, grad_w + dim_w,
                grad_w,
                [&xmabmx](T v) {return xmabmx * v;});
      return f;
    }
}; // Constrained1DField
} // namespace pift
#endif // PIFT_FIELD_HPP
