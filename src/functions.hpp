/* Some classes related to function approximations.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/16/2022
 *
 */

#ifndef PIFT_FUNCTIONS_HPP
#define PIFT_FUNCTIONS_HPP

#include <algorithm>
#include <cmath>


template <typename T>
class FunctionApproximation {
public:
  const int num_terms;
  const int dim;
  const T L;

  FunctionApproximation(int num_terms, const T& domain_length) :
  num_terms(num_terms),
  dim(1 + (num_terms - 1) * 2),
  L(domain_length)
  {}

  // Evaluate the function approximation at x
  // Parameters:
  // x -- The point on which to evaluate
  // w -- The weight vector of size dim
  T eval(const T& x, const T* w) const {
    T s = w[0];
    for(int i=1; i<num_terms; i++) {
      const T tmp = 2.0 * M_PI / L * i * x;
      s += w[i] * cos(tmp) + w[num_terms + i - 1] * sin(tmp);
    }
    return s;
  }

  void eval_and_prime(const T& x, const T* w, T& f, T& f_prime) const {
    f = w[0];
    f_prime = 0.0;
    for(int i=1; i<num_terms; i++) {
      const T omega = M_PI / L * i;
      const T omega_x = omega * x;
      const T cos_omega_x = cos(omega_x);
      const T sin_omega_x = sin(omega_x);
      const T omega_cos_omega_x = omega * cos_omega_x;
      const T omega_sin_omega_x = omega * sin_omega_x;
      f += w[i] * cos_omega_x + w[num_terms + i -1] * sin_omega_x;
      f_prime += (-w[i] * omega_sin_omega_x + w[num_terms + i - 1] * omega_cos_omega_x);
    }
  }

  // Evaluate the gradient of f with respect to w
  T eval_grad(
    const T& x, const T* w,
    T* grad_f
  ) const {
    T f = w[0];
    grad_f[0] = 1.0;
    for(int i=1; i<num_terms; i++) {
      const T omega = M_PI / L * i;
      const T omega_x = omega * x;
      const T cos_omega_x = cos(omega_x);
      const T sin_omega_x = sin(omega_x);
      const T omega_cos_omega_x = omega * cos_omega_x;
      const T omega_sin_omega_x = omega * sin_omega_x;
      f += w[i] * cos_omega_x + w[num_terms + i -1] * sin_omega_x;
      grad_f[i] = cos_omega_x;
      grad_f[num_terms + i - 1] = sin_omega_x;
    }
    return f;
  }

  // Evaluate the gradient of f with respect to w
  // and the gradient of f_prime with respect to w.
  void eval_grads(
    const T& x, const T* w,
    T& f, T* grad_f, T& f_prime, T* grad_f_prime
  ) const {
    f = w[0];
    grad_f[0] = 1.0;
    f_prime = 0.0;
    grad_f_prime[0] = 0.0;
    for(int i=1; i<num_terms; i++) {
      const T omega = M_PI / L * i;
      const T omega_x = omega * x;
      const T cos_omega_x = cos(omega_x);
      const T sin_omega_x = sin(omega_x);
      const T omega_cos_omega_x = omega * cos_omega_x;
      const T omega_sin_omega_x = omega * sin_omega_x;
      f += w[i] * cos_omega_x + w[num_terms + i -1] * sin_omega_x;
      grad_f[i] = cos_omega_x;
      grad_f[num_terms + i - 1] = sin_omega_x;
      f_prime += (-w[i] * omega_sin_omega_x + w[num_terms + i - 1] * omega_cos_omega_x);
      grad_f_prime[i] = -omega_sin_omega_x;
      grad_f_prime[num_terms + i - 1] = omega_cos_omega_x;
    }
  }

};

/* This is a function that is constrained on the boundary.
 */
template <typename T, typename FA>
class ConstrainedFunctionApproximation {
public:
  const FA& phi;
  const int dim;
  const T a;
  const T b;
  const T ya;
  const T yb;

  ConstrainedFunctionApproximation(
    const FA& phi,
    const T& a,
    const T& ya,
    const T& b,
    const T& yb
  ) :
  phi(phi),
  dim(phi.dim),
  a(a),
  b(b),
  ya(ya),
  yb(yb)
  {}

  T eval(const T& x, const T* w) const {
    const T xma = x - a;
    const T bmx = b - x;
    return bmx * ya + xma * yb + xma * bmx * phi.f(x, w);
  }

  void eval_and_prime(const T& x, const T* w, T& f, T& f_prime) const {
    const T xma = x - a;
    const T bmx = b - x;
    const T xmabmx = xma * bmx;
    const T bmxmxma = bmx - xma;
    phi.eval_and_prime(x, w, f, f_prime);
    f_prime = yb - ya + bmxmxma * f + xmabmx * f_prime;
    f = bmx * ya + xma * yb + xmabmx * f;
  }  

  T eval_grad(
    const T& x, const T* w,
    T* grad_f
  ) const {
    const T xma = x - a;
    const T bmx = b - x;
    const T xmabmx = xma * bmx;
    const T bmxmxma = bmx - xma;
    T f = phi.eval_grad(x, w, grad_f);
    f = bmx * ya + xma * yb + xmabmx * f;
    transform(grad_f, grad_f + dim,
              grad_f,
              [&xmabmx](T v) {return xmabmx * v;});
    return f;
  }

  void eval_grads(
    const T& x, const T* w,
    T& f, T* grad_f, T& f_prime, T* grad_f_prime
  ) const {
    const T xma = x - a;
    const T bmx = b - x;
    const T xmabmx = xma * bmx;
    const T bmxmxma = bmx - xma;
    phi.eval_grads(x, w, f, grad_f, f_prime, grad_f_prime);
    f_prime = yb - ya + bmxmxma * f + xmabmx * f_prime;
    f = bmx * ya + xma * yb + xmabmx * f;
    transform(grad_f_prime, grad_f_prime + dim,
              grad_f,
              grad_f_prime,
              [&bmxmxma, &xmabmx](T v1, T v2) {return bmxmxma * v2 + xmabmx * v1;});
    transform(grad_f, grad_f + dim,
              grad_f,
              [&xmabmx](T v) {return xmabmx * v;});
  }
};

template <typename T, typename FA>
inline ConstrainedFunctionApproximation<T, FA>
make_constrained_function_approximation(
    const FA& phi,
    const T& a, const T& ya,
    const T& b, const T& yb
) {
  return ConstrainedFunctionApproximation<T, FA>(phi, a, ya, b, yb);
}
#endif // PIFT_FUNCTIONS_HPP
