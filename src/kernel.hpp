// Some template classes for Gaussian random fields.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  1/8/2023
//

#ifndef PIFT_KERNEL_HPP
#define PIFT_KERNEL_HPP

#include <cmath>

namespace pift {

// A 1D covariance function
template<typename T, typename K>
class Covariance1D {
  protected:
    const K& k;

  public:
    Covariance1D(const K& k) : k(k) {}

    inline const K& get_kernel() const { return k; }

    // Calculates the cross covariance matrix between X1 and X2
    //
    // x1 is n1-dim vector
    // x2 is n2-dim vector
    // C is a n1 x n2 matrix
    inline void operator()(
        const T* x1, const int& n1,
        const T* x2, const int& n2,
        T* C
    ) const {
      for(int i=0; i<n1; i++)
        for(int j=0; j<n2; j++)
          C[i * n2 + j] = k(x1[i], x2[j]);
    }

    // Calculates the covariance matrix
    inline void operator()(
        const T* x, const int& n,
        T* C
    ) const {
      for(int i=0; i<n; i++) {
        C[i * n + i] = k(x[i], x[i]);
        for(int j=i+1; j<n; j++) {
          C[i * n + j] = k(x[i], x[j]);
          C[j * n + i] = C[i * n + j];
        }
      }
    }
}; // CovarianceFunction1D

// A squared exponential covariance in 1D
template<typename T>
class SquaredExponential1DKernel {
  protected:
    // The lengthscale
    const T ell;
    // The signal strength
    const T s;

  public:
    SquaredExponential1DKernel(const T& ell, const T& s) : ell(ell), s(s) {}
    
    inline T operator()(const T& x1, const T& x2) const {
      return s * std::exp(-0.5 * std::pow((x1 - x2) / ell, 2));
    }

    inline T integrate(const T& a, const T& b, const T& x) const {
      return s * ell * (
            std::erf((b - x) / std::sqrt(2.0) / ell) -
            std::erf((a - x) / std::sqrt(2.0) / ell)
            ) / std::sqrt(2.0 / M_PI);
    }

}; // SquaredExponentialCovariance
} // namespace pift
#endif // PIFT_KERNEL_HPP

