/* C++ implementation of pift.
 *
 * Author: Ilias Bilionis
 * Date: 12/14/2022
 */

#ifndef F
#define F float
#define STOF stof
#endif

#include <iostream>
#include <string>
#include <random>
#include <fstream>

#include "io.hpp"
#include "utils.hpp"
#include "functions.hpp"
#include "sgld.hpp"

using namespace std;


template <typename T, typename FA>
class Hamiltonian {
public:
  // A function approximation
  const FA& phi;
  // The number of physical parameters
  const int num_params;
  // The number of dimensions of the function approximator
  // It is set to be phi.dim
  const int dim;
  // The left side of the domain.
  // Set to phi.a
  const T a;
  // The right side of the domain
  // Set to phi.b
  const T b;
  // The volume of the domain
  // Set to b - a
  const T volume;
  // The number of points to use for unbiased estimator involving
  // expectations over x
  const int N;

  // Temporary space for dim variables
  T* grad_phi;
  T* grad_phi_prime;
  // A function that samples numbers uniformly between a and b.
  uniform_real_distribution<T>* unif;

  Hamiltonian(const FA& phi, const int& N) :
  phi(phi),
  dim(phi.dim),
  num_params(3),
  a(phi.a),
  b(phi.b),
  volume(phi.b - phi.a),
  N(N)
  {
    grad_phi = new T[phi.dim];
    grad_phi_prime = new T[phi.dim];
    unif = new uniform_real_distribution<T>(a, b);
  }

  ~Hamiltonian() {
    delete grad_phi;
    delete grad_phi_prime;
    delete unif;
  }

  T source_term(const T& x, const T* theta) const {
    return cos(4.0 * x);
  }

  inline T density(
    const T& x, const T& phi,
    const T& phi_prime, const T* theta
  ) const {
      return theta[0] * (
        0.5 * theta[1] * pow(phi_prime, 2)
        + 0.25 * theta[2] * pow(phi, 4)
        + phi * source_term(x, theta)
      );
  }

  inline T density_add_grad_theta(
      const T& x, 
      const T& phi, const T& phi_prime, const T* theta,
      T* grad_theta
  ) const {
    const T phi_4 = pow(phi, 4);
    const T phi_prime_2 = pow(phi, 2);
    const T tmp1 = 0.5 * phi_prime_2;
    const T tmp2 = 0.25 * phi_4;
    const T tmp3 = theta[1] * tmp1 + theta[2] * tmp2;
    grad_theta[0] += tmp3;
    grad_theta[1] += theta[0] * tmp1; 
    grad_theta[2] += theta[0] * tmp2;
    // TODO: This needs to be modified if the source term has
    // parameters
    return theta[0] * tmp3;
  }

  inline T dphi(
    const T& x, const T& phi,
    const T& phi_prime, const T* theta
  ) const {
    return theta[0] * (theta[2] * pow(phi, 3) + source_term(x, theta));
  }

  inline T dphi_prime(
    const T&x, const T& phi,
    const T& phi_prime, const T* theta
  ) const {
    return theta[0] * theta[1] * phi_prime;
  }

  T add_grad_w(
    const T& x, const T* w,
    const T* theta,
    T* grad_w_H
  ) {
    T phi_val;
    T phi_prime_val;
    phi.eval_grads(x, w, phi_val, grad_phi, phi_prime_val, grad_phi_prime);
    const T h = density(x, phi_val, phi_prime_val, theta);
    const T dh_dphi = dphi(x, phi_val, phi_prime_val, theta);
    const T dh_dphi_prime = dphi_prime(x, phi_val, phi_prime_val, theta);
    for(int i=0; i<dim; i++)
      grad_w_H[i] += dh_dphi * grad_phi[i] + dh_dphi_prime * grad_phi_prime[i];
    return h;
  }

  // Unbiased estimator of grad_w(H) at fixed w and theta.
  // Expectation is over spatial points.
  template <typename R>
  T unbiased_estimator_grad_w(
      const T* w,
      const T* theta,
      R& rng,
      T* grad_w_H
  ) {
    fill(grad_w_H, grad_w_H + dim, 0.0);
    T h = 0.0;
    for(int n=0; n<N; n++) {
      const T x = (*unif)(rng);
      h += add_grad_w(x, w, theta, grad_w_H);
    }
    h *= volume / N;
    scale(grad_w_H, dim, volume / N);
    return h;
  }

  // Unbiased estimator of grad_theta(H) at fixed w and theta.
  // Expectation is over spatial points.
  // TODO: Allow the user to select different number of spatial
  // points N for estimating the expectation.
  template <typename R>
  T unbiased_estimator_grad_theta(
      const T* w,
      const T* theta,
      R& rng,
      T* grad_theta
  ) {
    fill(grad_theta, grad_theta + num_params, 0.0);
    T h = 0.0;
    for(int n=0; n<N; n++) {
      const T x = (*unif)(rng);
      h += add_grad_theta(x, w, theta, grad_theta);
    }
    return h;
  }
};

template <typename T, typename FA>
class GaussianLikelihood {
public:
  const FA& phi;
  const int dim;
  const T sigma;
  const T sigma2;
  const T* x_obs;
  const T* y_obs;
  const int num_obs;
  const int M;
  T* grad_phi;
  uniform_int_distribution<int>* unif_int;

  GaussianLikelihood(
    const FA& phi,
    const T* x_obs,
    const T* y_obs,
    const int& num_obs,
    const T& sigma,
    const int& M
  ) :
  phi(phi),
  dim(phi.dim),
  sigma (sigma),
  sigma2 (pow(sigma, 2)),
  x_obs(x_obs),
  y_obs(y_obs),
  num_obs(num_obs),
  M(M)
  {
    grad_phi = new T[dim];
    unif_int = new uniform_int_distribution<int>(0, num_obs - 1);
  }

  ~GaussianLikelihood() {
    delete grad_phi;
    delete unif_int;
  }

  inline T minus_log_likelihood(const int& n, const T* w) {
    return 0.5 * (log(2.0 * M_PI * sigma) +
                  pow((phi->eval(x_obs[n], w) - y_obs[n]) / sigma, 2));
  }

  T add_grad_w(
    const int& n,
    const T* w,
    const T* theta,
    T* grad_w_minus_log_like
  ) {
    const T phi_n = phi.eval_grad(x_obs[n], w, grad_phi);
    const T std_err = (phi_n - y_obs[n]) / sigma2;
    // grad_w_minus_log_like = d_minus_log_like_d_phi * grad_w_phi
    for(int i=0; i<dim; i++)
      grad_w_minus_log_like[i] += std_err * grad_phi[i];
    return 0.5 * (log(2.0 * M_PI * sigma) +
                  pow(std_err, 2));
  }

  // TODO: Verify this unbiased estimator
  template <typename R>
  T unbiased_estimator_grad_w(
      const T* w,
      const T* theta,
      R& rng,
      T* grad_w
  ) {
    const T ratio = static_cast<T>(num_obs) / static_cast<T>(M);
    fill(grad_w, grad_w + dim, 0.0);
    T l = 0.0;
    for(int m=0; m<M; m++) {
      const int n = (*unif_int)(rng);
      l += add_grad_w(n, w, theta, grad_w);
    }
    l *= ratio;
    scale(grad_w, dim, ratio);
    return l;
  }
};

template <typename T, typename H, typename L>
class FullHamiltonian {
public:
  const int dim;
  H* h;
  L* l;
  T* grad_phi;

  FullHamiltonian(H& h, L& l) :
  dim(h.dim)
  {
    this->h = &h;
    this->l = &l;
    grad_phi = new T[dim];
  }

  ~FullHamiltonian() {
    delete grad_phi;
  }

  template <typename R>
  T unbiased_estimator_grad_w(
      const T* w,
      const T* theta,
      R& rng,
      T* grad_w
  ) {
    const T h_val = h->unbiased_estimator_grad_w(w, theta, rng, grad_w);
    const T l_val = l->unbiased_estimator_grad_w(w, theta, rng, grad_phi);
    for(int i=0; i<dim; i++)
      grad_w[i] += grad_phi[i];
    return h_val + l_val;
  }
};

// A class representing an unbiased estimator of the gradient of
// Hamiltonian with resepect to w at a fixed theta.
template <typename T, typename H, typename R>
class UnbiasedEstimatorOfGradWAtFixedTheta {
  public:
    // The Hamiltonian
    H& h;
    // The fixed theta
    T* theta;
    // A random number generator
    R& rng;
    // The dimension of w
    const int dim;

    UnbiasedEstimatorOfGradWAtFixedTheta(
        H& h, T* theta, R& rng
    ) :
      h(h), theta(theta), rng(rng), dim(h.dim)
    {}

    inline T operator()(const T* w, T* grad_w) {
      return h.unbiased_estimator_grad_w(w, theta, rng, grad_w);
    }
};


template <typename T, typename H, typename R>
inline
UnbiasedEstimatorOfGradWAtFixedTheta<T, H, R> make_unbiased_estimator_w(
    H& h, T* theta, R& rng
) {
  return UnbiasedEstimatorOfGradWAtFixedTheta<T, H, R>(h, theta, rng);
}


// A class representing an unbiased estimato of the gradient of the
// expectation of the Hamiltonian with respect to theta.
// The expectation is over w and it is estimated using samples from SGLD
template <typename T, typename H, typename HA, typename R>
class UnbiasedEstimatorOfGradTheta {
public:
  // The Hamiltonian to sample from
  H& h;
  // The Hamiltonian the gradient of which we need to take
  HA& ha;
  // The random number generator
  R& rng;
  // The dimension of w
  const int dim;
  // The number of parameters
  const int num_params;
  // The number of different chains
  const int num_chains;
  // Number of warmup steps
  const int num_warmup;
  // Number of total steps
  const int num_samples;
  // Number of thinning steps
  const int num_thinning;
  // Number of sampling bursts = num_samples / num_thinning
  const int num_bursts;
  // The parameters of SGLD
  const SGLDParams<T>& sgld_params;
  // Unbiased estimator of grad_w (H) so that we can sample
  UnbiasedEstimatorOfGradWAtFixedTheta<T, H, R>& ue_grad_w;
  // Standard deviation used to initialize the ws
  const T init_w_sigma;
  // Whether or not to reinitialize the samples of w
  const bool reinitialize_ws;
  // Space we need for storing grad_w_H
  T* grad_w_H;
  T* ws;
  T* tmp_grad_theta;

  UnbiasedEstimatorOfGradTheta(
      H& h, HA& ha, R& rng,
      SGLDParams<T>& sgld_params,
      const int& num_chains,
      const int& num_warmup, const int& num_samples, const int& num_thinning,
      const T& init_w_sigma=1.0,
      const bool& reinitialize_ws=false
  ) :
    h(h), ha(ha), rng(rng), dim(h.dim), num_params(h.num_params),
    num_chains(num_chains),
    num_warmup(num_warmup), num_samples(num_samples), num_thinning(num_thinning),
    init_w_sigma(init_w_sigma),
    reinitialize_ws(reinitialize_ws),
    sgld_params(sgld_params),
    ue_grad_w(h, NULL, rng),
    num_bursts(num_samples / num_thinning)
  {
    grad_w_H = new T[dim];
    ws = new T[num_chains * dim];
    tmp_grad_theta = new T[num_params];
  }

  ~UnbiasedEstimatorOfGradTheta() {
    delete grad_w_H;
    delete ws;
    delete tmp_grad_theta;
  }

  inline void initialize_chains() {
    normal_distribution<T> norm(0, init_w_sigma);
    R rng = this->rng;
    generate(ws, ws + num_chains * dim, [&norm, &rng]() {return norm(rng);});
  }

  T operator()(const T* theta, T* grad_theta) {
    if (reinitialize_ws)
      initialize_chains();
    T h = 0.0;
    fill(grad_theta, grad_theta + num_params, 0.0);
    // Make sure the unbiased estimator sees the right theta
    ue_grad_w.theta = theta;
    // TODO: Exploit parallelization opportunities
    // Loop over chains
    for(int c=0; c<num_chains; c++) {
      const T* w = ws + c * dim;
      // Loop over bursts
      for(int b=0; b<num_bursts; b++) {
        // Sample w num_thinning times
        sgld(ue_grad_w, w, rng, num_thinning, grad_w_H, sgld_params);
        // Now w contains the sample
        // Get the gradient with respect to theta
        h += ha.unbiased_estimator_grad_theta(w, theta, rng, tmp_grad_theta);
        for(int i=0; i<num_params; i++)
          grad_theta[i] += tmp_grad_theta[i];
      }
    }
    // Divide with the total number of samples
    scale(grad_theta, 1.0 / (num_chains * num_bursts));
    return h;
  }
};


template <typename T, typename H, typename R>
void unit_test(H& h, R& rng) {
  const int N = 100;
  const int dim = h.dim;
  T x[N];
  T w[dim];
  normal_distribution<T> norm(0, 1);

  // Some points to evaluate things on
  linspace(h.a, h.b, N, x);
  savetxt(x, N, "src/unit_test_x.csv");

  // Some random weights
  generate(w, w + dim, [&norm, &rng]() {return norm(rng);});
  savetxt(w, dim, "src/unit_test_w.csv");

  // The parameters to use
  T theta[] = {10.0, 0.1, 1.0};
  savetxt(theta, 3, "src/unit_test_theta.csv");

  // Evaluate the parameterization and its gradients
  T f[N];
  T grad_f[N * dim];
  T f_prime[N];
  T grad_f_prime[N * dim];
  for(int n=0; n<N; n++)
    h.phi.phi.eval_grads(
      x[n], w,
      f[n], grad_f + n * dim,
      f_prime[n], grad_f_prime + n * dim
    );
  savetxt(f, N, "src/unit_test_psi.csv");
  savetxt(grad_f, N, dim, "src/unit_test_grad_w_psi.csv");
  savetxt(f_prime, N, "src/unit_test_psi_prime.csv");
  savetxt(grad_f_prime, N, dim, "src/unit_test_grad_w_psi_prime.csv");

  // Evaluate the constrained parameterization and its gradients
  for(int n=0; n<N; n++)
    h.phi.eval_grads(
      x[n], w,
      f[n], grad_f + n * dim,
      f_prime[n], grad_f_prime + n * dim
    );
  savetxt(f, N, "src/unit_test_phi.csv");
  savetxt(grad_f, N, dim, "src/unit_test_grad_w_phi.csv");
  savetxt(f_prime, N, "src/unit_test_phi_prime.csv");
  savetxt(grad_f_prime, N, dim, "src/unit_test_grad_w_phi_prime.csv");

  // Evaluate the Hamiltonian
  T grad_w_H[N * dim];
  fill(grad_w_H, grad_w_H + N * dim, 0.0);
  for(int n=0; n<N; n++)
    f[n] = h.add_grad_w(
      x[n], w,
      theta,
      grad_f + n * dim
    );
  savetxt(f, N, "src/unit_test_H.csv");
  savetxt(grad_f, N, dim, "src/unit_test_grad_w_H.csv");
}


template <typename T, typename H, typename R>
void unit_test_sample_w(
    H& h, R& rng, const int num_samples,
    const SGLDParams<T>& sgld_params
) {
  F theta[] = {10000.0, 0.1, 1.0};
  F w[h.dim];
  fill(w, w + h.dim, 0.0);
  F grad_w_H[h.dim];

  auto ue_grad_w = make_unbiased_estimator_w(h, theta, rng);

  sgld(ue_grad_w, w, rng, num_samples, sgld_params);
}


int main(int argc, char* argv[]) {
  if (argc != 7) {
    cout << "Usage:\n\t" + string(argv[0]) + " <N> <alpha> <beta> <gamma> <maxit> <out_file>" << endl;
    exit(1);
  }

  using FA = FunctionApproximation<F>;
  using CFA = ConstrainedFunctionApproximation<F, FA>;
  using H = Hamiltonian<F, CFA>;
  using G = GaussianLikelihood<F, CFA>;
  using FH = FullHamiltonian<F, H, G>;

  const int N = stoi(argv[1]);
  const F alpha = STOF(argv[2]);
  const F beta = STOF(argv[3]);
  const F gamma = STOF(argv[4]);
  const int num_samples = stoi(argv[5]);
  const string out_file = argv[6];
  const F sigma = 0.01;
  const int M = 1;

  //random_device rand_dev;
  mt19937 rng;

  FA psi(4, 1.0);
  CFA phi(psi, 0.0, 0.0, 1.0, 0.0);
  
  H h(phi, N);

  // unit_test(h, rng);

  SGLDParams<F> sgld_params;
  sgld_params.alpha = alpha;
  sgld_params.beta = beta;
  sgld_params.gamma = gamma;
  sgld_params.save_to_file = true;
  sgld_params.out_file = out_file + "_prior.csv";
  unit_test_sample_w(h, rng, num_samples, sgld_params);

  // Read observations
  string x_file("src/x_obs.csv");
  auto x_obs = loadtxtvec<F>(x_file);
  string y_file("src/y_obs.csv");
  auto y_obs = loadtxtvec<F>(y_file);

  // Make the likelihood
  G l(
    phi,
    x_obs.data(), y_obs.data(), x_obs.size(),
    sigma,
    M
  );

  //// Make the full Hamiltonian
  FH fh(h, l);

  sgld_params.out_file = out_file + "_post.csv";
  unit_test_sample_w(fh, rng, num_samples, sgld_params);
}
