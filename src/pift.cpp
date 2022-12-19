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
  T beta;
  // A function that samples numbers uniformly between a and b.
  uniform_real_distribution<T>* unif;

  Hamiltonian(const FA& phi, const int& N) :
  phi(phi),
  dim(phi.dim),
  num_params(2),
  a(phi.a),
  b(phi.b),
  volume(phi.b - phi.a),
  N(N)
  {
    beta = 1000.0;
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
      // return theta[0] * (
      return beta * (
        0.5 * theta[0] * pow(phi_prime, 2)
        + 0.25 * theta[1] * pow(phi, 4)
        + phi * source_term(x, theta)
      );
  }

  inline T add_grad_theta(
      const T& x, 
      const T* w,
      const T* theta,
      T* grad_theta
  ) const {
    T phi_val, phi_prime_val;
    phi.eval_and_prime(x, w, phi_val, phi_prime_val);
    const T phi_4 = pow(phi_val, 4);
    const T phi_prime_2 = pow(phi_prime_val, 2);
    const T tmp1 = 0.5 * phi_prime_2;
    const T tmp2 = 0.25 * phi_4;
    const T tmp3 = theta[1] * tmp1 + theta[2] * tmp2;
    //grad_theta[0] += tmp3;
    //grad_theta[1] += theta[0] * tmp1; 
    //grad_theta[2] += theta[0] * tmp2;
    grad_theta[0] += beta * tmp1; 
    grad_theta[1] += beta * tmp2;
    // TODO: This needs to be modified if the source term has
    // parameters
    //return theta[0] * tmp3;
    return beta * tmp3;
  }

  inline T dphi(
    const T& x, const T& phi,
    const T& phi_prime, const T* theta
  ) const {
    //return theta[0] * (theta[2] * pow(phi, 3) + source_term(x, theta));
    return beta * (theta[1] * pow(phi, 3) + source_term(x, theta));
  }

  inline T dphi_prime(
    const T&x, const T& phi,
    const T& phi_prime, const T* theta
  ) const {
    //return theta[0] * theta[1] * phi_prime;
    return beta * theta[0] * phi_prime;
  }

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
    // prior on theta is p(theta) = 1/theta
    // log p(theta) = - log(theta)
    // nabla log p(theta) = - 1 / theta
    // -nabla log p(theta) = 1 / theta
    // So, we need -nabla log p(theta) = -nabla (- log theta) = 1 / theta 
    for(int i; i<num_params; i++)
      grad_theta[i] = 1.0 / theta[i];
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
    const T* theta;
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


template <typename T>
struct UEThetaParams {
  UEThetaParams() :
    num_chains(1),
    num_init_warmup(10000),
    num_per_it_warmup(1),
    num_bursts(1),
    num_thinning(1),
    init_w_sigma(1.0),
    reinitialize_ws(false),
    save_to_file(false),
    out_file("ue_theta.csv"),
    save_freq(10),
    disp(true),
    disp_freq(100),
    sgld_params(SGLDParams<T>())
  {}
  // The number of chains
  int num_chains;
  // The number of initial warmup steps
  int num_init_warmup;
  // The number of warmup steps per iteration
  int num_per_it_warmup;
  // The number of bursts (aka number of (almost) independent samples)
  int num_bursts;
  // The number of samples between bursts
  int num_thinning;
  // The variance for initializing sigma
  T init_w_sigma;
  // Whether or not to reinitialize the ws on every iteration
  bool reinitialize_ws;
  // Whether or not to save to file the results
  bool save_to_file;
  // How often to write on the file
  int save_freq;
  // The file to write
  string out_file;
  // Whether or not to display something on the screen
  bool disp;
  // The display frequency
  int disp_freq;
  // The parameters used in SGLD
  SGLDParams<T> sgld_params; 
};


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
  // The parameters structure
  UEThetaParams<T>& params;
  // Unbiased estimator of grad_w (H) so that we can sample
  UnbiasedEstimatorOfGradWAtFixedTheta<T, H, R>* ue_grad_w;
  // Space we need for storing grad_w_H
  T* grad_w_H;
  T* ws;
  T* tmp_grad_theta;
  normal_distribution<T>* norm;

  UnbiasedEstimatorOfGradTheta(
      H& h, HA& ha, R& rng,
      UEThetaParams<T>& params
  ) :
    h(h), ha(ha), rng(rng), dim(h.dim), num_params(ha.num_params),
    params(params)
  {
    grad_w_H = new T[dim];
    ws = new T[params.num_chains * dim];
    tmp_grad_theta = new T[num_params];
    ue_grad_w = new UnbiasedEstimatorOfGradWAtFixedTheta<T, H, R>(h, NULL, rng);
    norm = new normal_distribution<T>(0, 1);
    initialize_chains();
  }

  ~UnbiasedEstimatorOfGradTheta() {
    delete grad_w_H;
    delete ws;
    delete tmp_grad_theta;
    delete ue_grad_w;
    delete norm;
  }

  inline void initialize_chains() {
    normal_distribution<T> norm(0, params.init_w_sigma);
    R rng = this->rng;
    generate(ws, ws + params.num_chains * dim, [&norm, &rng]() {return norm(rng);});
  }

  inline void warmup(const T* theta) {
    ue_grad_w->theta = theta;
    for(int c=0; c<params.num_chains; c++) {
      T* w = ws + c * dim;
      sgld(*ue_grad_w, w, rng, params.num_init_warmup, grad_w_H, *norm, params.sgld_params);
    }
  }

  T operator()(const T* theta, T* grad_theta) {
    if (params.reinitialize_ws)
      initialize_chains();
    T h = 0.0;
    fill(grad_theta, grad_theta + num_params, 0.0);
    // Make sure the unbiased estimator sees the right theta
    ue_grad_w->theta = theta;
    // TODO: Exploit parallelization opportunities
    // Loop over chains
    for(int c=0; c<params.num_chains; c++) {
      T* w = ws + c * dim;
      // Do the warmup
      sgld(*ue_grad_w, w, rng, params.num_per_it_warmup, grad_w_H, *norm, params.sgld_params);
      // Loop over bursts
      for(int b=0; b<params.num_bursts; b++) {
        // Sample w num_thinning times
        sgld(*ue_grad_w, w, rng, params.num_thinning, grad_w_H, *norm, params.sgld_params);
        // Now w contains the sample
        // Get the gradient with respect to theta
        h += ha.unbiased_estimator_grad_theta(w, theta, rng, tmp_grad_theta);
        for(int i=0; i<num_params; i++)
          grad_theta[i] += tmp_grad_theta[i];
      }
    }
    // Divide with the total number of samples
    const T lambda = T(1.0) / (params.num_chains * params.num_bursts);
    scale(grad_theta, num_params, lambda);
    h *= lambda;
    return h;
  }
};

template <typename T, typename H, typename FH, typename R>
class UnbiasedEstimatorGradThetaMinusLogPosterior {
public:
  UnbiasedEstimatorOfGradTheta<T,H,H,R>* ue_prior;
  UnbiasedEstimatorOfGradTheta<T,FH,H,R>* ue_post;
  T* grad_theta_prior;
  int dim;
  UnbiasedEstimatorGradThetaMinusLogPosterior(
      H& h, FH& fh, R& rng,
      UEThetaParams<T>& prior_params,
      UEThetaParams<T>& post_params
  ) : dim(h.num_params)
  {
    ue_prior = new UnbiasedEstimatorOfGradTheta<T,H,H,R>(h,h,rng, prior_params);
    ue_post = new UnbiasedEstimatorOfGradTheta<T,FH,H,R>(fh,h,rng, post_params);
    grad_theta_prior = new T[dim];
  }

  ~UnbiasedEstimatorGradThetaMinusLogPosterior() {
    delete ue_prior;
    delete ue_post;
    delete grad_theta_prior;
  }

  T operator()(const T* theta, T* grad_theta) {
    const T h_prior = (*ue_prior)(theta, grad_theta_prior);
    const T h_post = (*ue_post)(theta, grad_theta);
    for(int i=0; i<dim; i++)
      grad_theta[i] -= grad_theta_prior[i];
    return h_post - h_prior;
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
  // TYPE ALIASES
  // The function approximation
  using FA = FunctionApproximation<F>;
  // A function approximation that satisfies the boundary conditions
  using CFA = ConstrainedFunctionApproximation<F, FA>;
  // The Hamiltonian
  using H = Hamiltonian<F, CFA>;
  // The likelihood
  using G = GaussianLikelihood<F, CFA>;
  // The posterior hamiltonian (or Hamiltonian minus log likelihood)
  using FH = FullHamiltonian<F, H, G>;
  // A random number generator
  using R = mt19937;
  // An unbiased estimator of the grad_theta H, expectation over the prior
  using UEPR = UnbiasedEstimatorOfGradTheta<F, H, H, R>;
  // An unbiased estimator of the grad_theta H, expectation over the posterior
  using UEPS = UnbiasedEstimatorOfGradTheta<F, FH,H, R>; 
  using UEGTheta = UnbiasedEstimatorGradThetaMinusLogPosterior<F, H, FH, R>;

  const int N = stoi(argv[1]);
  const F alpha = STOF(argv[2]);
  const F beta = STOF(argv[3]);
  const F gamma = STOF(argv[4]);
  const int num_samples = stoi(argv[5]);
  const string out_file = argv[6];
  const F sigma = 0.01;
  const int M = 1;

  //random_device rand_dev;
  R rng;

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
  //unit_test_sample_w(h, rng, num_samples, sgld_params);

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
  //unit_test_sample_w(fh, rng, num_samples, sgld_params);

  // Test the unbiased estimator of grad_theta that samples from the prior
  UEThetaParams<F> prior_params;
  UEThetaParams<F> post_params;
  UEPR uepr(h, h, rng, prior_params);
  prior_params.num_init_warmup = 100000;
  prior_params.num_bursts = 1;
  prior_params.num_per_it_warmup = 1;
  prior_params.num_thinning = 1;
  UEPS ueps(fh, h, rng, post_params);
  post_params.num_init_warmup = 100000;
  post_params.num_bursts = 1;
  prior_params.num_per_it_warmup = 1;
  post_params.num_thinning = 1;
  prior_params.sgld_params.save_to_file = false;
  prior_params.sgld_params.disp = false;
  post_params.sgld_params.save_to_file = false;
  post_params.sgld_params.disp = false;
  UEGTheta ue_grad_theta(h, fh, rng, prior_params, post_params);

  //F theta[] = {10.0, 0.01, 1.0};
  F theta[] = {1.0, 1.0};
  F grad_theta[h.num_params];

  SGLDParams<F> sgld_params_theta;
  sgld_params_theta.alpha = 1e-4;
  sgld_params_theta.disp_freq = 100000;
  sgld_params_theta.save_to_file = true;
  sgld_params_theta.out_file = "src/theta.csv";
  sgld_params_theta.save_freq = 10000;
  normal_distribution<F> norm(0,1);
  sgld(ue_grad_theta, theta, rng, 100000000, sgld_params_theta); 
}

