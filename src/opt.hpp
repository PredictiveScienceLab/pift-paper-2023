/* Some functions related to optimization.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/16/2022
 *
 */

#ifndef PIFT_OPT_HPP
#define PIFT_OPT_HPP

#include <random>
#include <string>
#include <cmath>

using namespace std;


template <typename T>
struct SGLDParams {

  SGLDParams() :
    alpha(1e-3),
    beta(0.0),
    gamma(0.6),
    maxit(1000),
    save_to_file(false),
    out_file("sgld_deafult_out.csv"),
    save_freq(10),
    disp(true),
    disp_freq(100)
  {}

  T alpha;
  T beta;
  T gamma;
  int maxit;
  bool save_to_file;
  string out_file;
  int save_freq;
  bool disp;
  int disp_freq;
};


// Implementation of Stochastic Gradient Lagevin Dynamics
template <typename T, typename UE, typename R>
void sgld(UE& ue_func,
          T* w,
          R& rng,
          const SGLDParams<T>& params = SGLDParams<T>()
) {
  const int& dim = ue_func.dim;
  T grad_w_H[dim];
  normal_distribution<T> norm(0, 1);
  ofstream of;
#ifndef DISABLE_SGLD_SAVE
  if (params.save_to_file)
    of.open(params.out_file);
#endif
  for(int it=0; it<params.maxit; it++) {
    const T epsilon = params.alpha / pow((params.beta + it + 1), params.gamma);
    const T sqrt_2_epsilon = sqrt(2.0 * epsilon);
    const T h_val = ue_func(w, grad_w_H);
    for(int i=0; i<dim; i++)
      w[i] += (-epsilon * grad_w_H[i] + sqrt_2_epsilon * norm(rng));
#ifndef DISABLE_SGLD_SAVE
    if (params.save_to_file && it % params.save_freq == 0) {
      of << h_val << " ";
      cout_vec(w, dim, of);
    }
#endif
#ifndef DISABLE_SGLD_DISP
    if (params.disp && it % params.disp_freq == 0)
      cout_vec(w, dim, to_string(it) + ": " + to_string(h_val) + " ");
  }
#endif
#ifndef DISABLE_SGLD_SAVE
  if (params.save_to_file)
    of.close();
#endif
}
#endif // PIFT_OPT_HPP