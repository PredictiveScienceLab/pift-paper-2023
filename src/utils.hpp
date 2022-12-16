/* Some utility functions.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/16/2022
 *
 */


template <typename T>
void linspace(const T& a, const T& b, const int& N, T* x) {
  const T dx = (b - a) / static_cast<T>(N - 1);
  for(int i=0; i <N; i++)
    x[i] = i * dx;
}

template <typename T>
inline void scale(const T* x, const int& N, const T& a, T* out) {
  for(int i=0; i<N; i++)
    out[i] = a * x[i];
}

template <typename T>
inline void scale(T* x, const int& N, const T& a) {
  scale<T>(x, N, a, x);
}


