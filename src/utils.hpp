// Some utility functions.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/16/2022

#ifndef PIFT_UTILS_HPP
#define PIFT_UTILS_HPP

namespace pift {

// Spans the space from [a, b] with n points.
template <typename T>
void linspace(const T& a, const T& b, const int& n, T* x) {
  const T dx = (b - a) / static_cast<T>(n - 1);
  for(int i=0; i <n; i++)
    x[i] = i * dx;
}

// Scale a vector
template <typename T>
inline void scale(const T* x, const int& n, const T& a, T* out) {
  for(int i=0; i<n; i++)
    out[i] = a * x[i];
}

// Scale a vector in place
template <typename T>
inline void scale(T* x, const int& n, const T& a) {
  scale<T>(x, n, a, x);
}

} // namespace pift

#endif // PIFT_UTILS_HPP
