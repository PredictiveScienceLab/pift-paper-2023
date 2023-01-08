// Tests for covariance functions

#include <iostream>
#include <fstream>
#include <random>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>

#include "pift.hpp"

using RNG = std::mt19937;
using Kernel = pift::SquaredExponential1DKernel<double>;
using Covariance = pift::Covariance1D<double, Kernel>;
using Domain = pift::UniformRectangularDomain<double, RNG>;
using KLE = pift::KLE<Domain,Kernel>;

int main(int argc, char* argv[]) {

  const double ell = 0.3;
  const double s = 1.0;
  const double a = 0.0;
  const double b = 1.0;

  Kernel k(ell, s);
  Covariance c(k);

  const int n = 10;
  double x[n];
  pift::linspace(a, b, n, x);
  
  double C[n * n];
  
  std::cout << "*" << std::endl;
  c(x, n, C);
  pift::cout_mat(C, n, n);

  std::cout << "*" << std::endl;
  c(x, n, x, n, C);
  pift::cout_mat(C, n, n);

  std::cout << "*" << std::endl;
  gsl_matrix* Cp = gsl_matrix_alloc(n, n);
  c(x, n, Cp->data);
  pift::cout_mat(Cp->data, n, n);

  // doubleind eigenvalues
  gsl_eigen_symmv_workspace* work = gsl_eigen_symmv_alloc(n);
  gsl_vector* eval = gsl_vector_alloc(n);
  gsl_matrix* evec = gsl_matrix_alloc(n, n);
 
  gsl_eigen_symmv(Cp, eval, evec, work);

  // eigenvalues are not guaranteed to be ordered
  std::cout << "Eigenvalues:" << std::endl;
  pift::cout_vec(eval->data, eval->size);

  // eigenvectors are in the columns, they are orthogonal and normalized
  std::cout << "Eigenvectors:" << std::endl;
  pift::cout_mat(evec->data, n, n);

  std::cout << "v1:" << std::endl;
  for(int i=0; i<n; i++)
    std::cout << gsl_matrix_get(evec, i, 0) << std::endl;

  // Test normalization
  double r = 0.0;
  for(int i=0; i<n; i++)
    r += std::pow(gsl_matrix_get(evec, i, 0), 2);
  std::cout << "norm(v1) = " << std::sqrt(r) << std::endl;

  gsl_matrix_free(Cp);
  gsl_eigen_symmv_free(work);
  gsl_vector_free(eval);
  gsl_matrix_free(evec);

  // Test the Karhunen-Loeve expansion
  RNG rng;
  std::vector<std::vector<double>> bounds{{0.0, 1.0}};
  Domain domain(bounds, rng);

  const double mean_value = 0.25 * std::sin(4.0);
  const int num_terms = 10;
  const int nq = 100;
  KLE kle(domain, k, num_terms, mean_value, nq);

  std::cout << "KLE" << std::endl;
  std::cout << "Eigenvalues: " << std::endl;
  pift::cout_vec(kle.get_eval(), nq);
  std::cout << "Order: " << std::endl;
  pift::cout_vec(kle.get_idx());
  std::cout << "Energy explained: " << kle.get_explained_energy() << std::endl;

  const int ns = 100;
  double xs[ns];
  pift::linspace(a, b, ns, xs);
 
  // Test the eigen vectors
  std::ofstream eigen_out("eigen.csv");
  for(int i=0; i<ns; i++) {
    eigen_out << xs[i];
    for(int j=0; j<kle.get_dim_w(); j++)
      eigen_out << " " << (
          kle.eigen_func(j, xs[i]) / kle.get_eval()[kle.get_idx()[j]] 
          * std::sqrt(nq)
          );
    eigen_out << std::endl;
  } 
  eigen_out.close();

  // Test samples
  std::normal_distribution<double> norm(0,1);
  double w[num_terms];
  std::ofstream samples_out("samples.csv");
  const int num_samples=10;
  for(int s=0; s<num_samples; s++) {
    for(int j=0; j<num_terms; j++)
      w[j] = norm(rng);
    for(int i=0; i<ns; i++)
      samples_out << kle(xs + i, w) << " ";
    samples_out << std::endl;
  }
  samples_out.close();

  return 0;
}
