// Debugging the constrained mean functionality

#include <cmath>
#include <random>
#include <vector>

#include "pift.hpp"

using RNG = std::mt19937;
using F = float;
using Domain = pift::UniformRectangularDomain<F, RNG>;
using Field = pift::RBF1DField<F, Domain>;
using CMField = pift::ConstrainedMeanField<F, Field, Domain>;

int main(int argc, char* argv[]) {
  RNG rng;

  std::vector<std::vector<F>> bounds{{0.0, 1.0}};

  Domain domain(bounds, rng);

  const int num_centers = 10;
  std::vector<F> centers(num_centers);
  pift::linspace(domain.a(0), domain.b(0), num_centers, centers.data());
  const F ell = 0.1;
  Field psi(centers, ell);

  const F f0 = 0.25 * std::sin(4.0);
  CMField phi(psi, domain, f0);

  F w[phi.get_dim_w()];
  std::normal_distribution<F> norm(0.0, 1.0);
  std::generate(w, w + phi.get_dim_w(), [&norm, &rng]() {return norm(rng);});

  std::cout << std::setprecision(2);
  // Test if the mean is correct
  const int n = 10000;
  F s = F(0.0);
  F xs[n];
  pift::linspace<F>(F(0.0), F(1.0), n, xs);
  for(int i=0; i<n; i++)
    s += phi(xs + i, w);
  std::cout << "compare " << (s / n) << " with " << f0 << std::endl;

  // Test the gradient with respect to w
  F grad_w[phi.get_dim_w()];
  const int m = 10;
  assert(m <= n);
  pift::linspace<F>(F(0.0), F(1.0), m, xs);
  const F h = F(0.01);
  for(int i=0; i<m; i++) {
    const F f = phi.eval_grad(xs + i, w, grad_w);
    std::cout << "***: " << xs[i] << std::endl;
    pift::cout_vec(grad_w, phi.get_dim_w(), std::cout, "grad_w: ");
    std::cout << "grad_w: ";
    for(int j=0; j<phi.get_dim_w(); j++) {
      w[j] += h;
      const F fh = phi.eval_grad(xs + i, w, grad_w);
      std::cout << (fh - f) / h << " ";
      w[j] -= h;
    }
    std::cout << std::endl;
  }
}
