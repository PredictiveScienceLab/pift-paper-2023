// Karhunen-Loeve expansion of Gaussian random field
//
// doubleODO: doublehis relies on GSL for eigenvalues and it can only work with double
// precision. 
//
// doubleODO: It only works in 1D.
//
// Author:
//  Ilias Bilionis
// 
// Date:
//  1/8/2023
//

#ifndef PIFdouble_KLE_HPP
#define PIFdouble_KLE_HPP

#include <exception>
#include <vector>
#include <numeric>
#include <algorithm>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

#include "field.hpp"
#include "utils.hpp"
#include "kernel.hpp"

namespace pift {

template<typename D, typename K>
class KLE : public ParameterizedField<double,D> {
  private:
    // The collocation points
    double* xq;
    double* sqrt_eval;
    double* mu;

    // The eigenvalues
    gsl_vector* eval;
    // The eigenvectors
    gsl_matrix* evec;
    // The order of the eigenvalues
    std::vector<int> idx;

  protected:
    const double mean_value;
    const int num_terms;
    const D& domain;
    const K& kernel;

  public:
    KLE(
       const D& domain,
       const K& kernel,
       const int& num_terms,         // Number of terms in the KLE
       const double& mean_value=0.0, // Mean value
       const int& nq=100             // Number of quadrature points for Nystrom
    ) :
      ParameterizedField<double,D>(1, num_terms, 1),
      domain(domain),
      kernel(kernel),
      num_terms(num_terms),
      mean_value(mean_value)
    {
      assert(domain.get_dim() == 1);
      assert(nq > 0);

      xq = new double[nq];
      pift::linspace(domain.a(0), domain.b(0), nq, xq);

      // Evaluate covariance matrix on collocation points
      gsl_matrix* cov_mat = gsl_matrix_alloc(nq, nq);
      Covariance1D<double, K> covariance(kernel);
      covariance(xq, nq, cov_mat->data);
      
      // Find eigenvalues and eigenvectors
      eval = gsl_vector_alloc(nq);
      evec = gsl_matrix_alloc(nq, nq);
      gsl_eigen_symmv_workspace* work = gsl_eigen_symmv_alloc(nq);
      gsl_eigen_symmv(cov_mat, eval, evec, work);
      gsl_eigen_symmv_free(work);
      gsl_matrix_free(cov_mat);

      // Sort the eigenvalues
      idx.resize(nq);
      iota(idx.begin(), idx.end(), 0);
      std::stable_sort(idx.begin(), idx.end(),
          [eval=this->eval](int i1, int i2) {
            return gsl_vector_get(eval, i1) > gsl_vector_get(eval, i2);
          }
      );
      sqrt_eval = new double[num_terms];
      for(int i=0; i<num_terms; i++)
        sqrt_eval[i] = std::sqrt(gsl_vector_get(eval, idx[i]));

      mu = new double[num_terms];
      for(int j=0; j<num_terms; j++)
        mu[j] = integrate_eigen_func(j);
    }

    ~KLE() {
      delete xq;
      delete sqrt_eval;
      delete mu;
      gsl_matrix_free(evec);
      gsl_vector_free(eval);
    }

    // Get the eigenvalues
    const double* get_eval() const { return eval->data; }

    // Get the eigenvectors
    const double* get_evec() const { return evec->data; }

    // Get the order of the eigenvalues
    const std::vector<int>& get_idx() const { return idx; }

    // Get the percent of the energy captured by the approximation
    double get_explained_energy() const {
      double total = 0.0;
      for(int i=0; i<eval->size; i++)
        total += gsl_vector_get(eval, i);
      double field = 0.0;
      for(int i=0; i<num_terms; i++)
        field += gsl_vector_get(eval, idx[i]);
      return field / total;
    }

    // Get the j-th eigen function (not normalized)
    inline double eigen_func(const int& j, const double& x) const {
      double s = 0.0;
      for(int i=0; i<eval->size; i++)
        s += kernel(x, xq[i]) * gsl_matrix_get(evec, i, idx[j]);
      return s;
    }

    inline double integrate_eigen_func(const int& j) const {
      double s = 0.0;
      for(int i=0; i<eval->size; i++)
        s += kernel.integrate(domain.a(0), domain.b(0), xq[i]) *
            gsl_matrix_get(evec, i, idx[j]);
      return s;
    }

    double operator()(const double* x, const double* w) const {
      double s = mean_value;
      for(int j=0; j<num_terms; j++)
        s += w[j] * eigen_func(j, x[0]) / sqrt_eval[j];
      return s;
    }

    void operator()(const double* x, const double* w, double* prolong) const {
      throw std::runtime_error("Not implemented.");
    }

    void operator()(
        const double* x, const double* w, double* prolong, double* grad_w_prolong
    ) const {
      throw std::runtime_error("Not implemented.");
    }

    double eval_grad(const double* x, const double* w, double* grad_w) const {
      double s = mean_value;
      for(int j=0; j<num_terms; j++) {
        grad_w[j] = eigen_func(j, x[0]) / sqrt_eval[j];
        s += w[j] * grad_w[j];
      } 
      return s;
    }

    double integrate(const D& domain, const double* w) const {
      double s = mean_value;
      for(int j=0; j<num_terms; j++)
        s += w[j] * mu[j] / sqrt_eval[j];
      return s;
    }

    double integrate(const D& domain, const double* w, double* grad_w) const {
      double s = mean_value;
      for(int j=0; j<num_terms; j++) {
        grad_w[j] = mu[j] / sqrt_eval[j];
        s += w[j] * grad_w[j];
      }
      return s;
    }
}; // ParameterizedField
} // namespace pift
#endif // PIFdouble_KLE_HPP
