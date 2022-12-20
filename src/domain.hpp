// Implements a uniform domain class.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/19/2022
//
//

#ifndef PIFT_DOMAIN_HPP
#define PIFT_DOMAIN_HPP

#include <random>
#include <vector>

namespace pift {

template<typename T, typename R>
class UniformRectangularDomain {
  protected:
    std::vector<T> bounds;
    T volume;
    const int dim;
    std::uniform_real_distribution<T>* unif;
    R& rng;

    inline T calculate_volume() const {
      T v = 1.0;
      for(int i=0; i<get_dim(); i++)
        v *= bounds[2 * i + 1] - bounds[2 * i];
      return v;
    }

  public:
    // The bounds of the box x_(i=1)^d [a_i, b_i] written in a 1D pointer as
    // [a_1, b_1, a_2, b_2, ..., a_d, b_d]
    UniformRectangularDomain(const T* bounds, const int& dim, R& rng) :
      dim(dim),
      rng(rng)
    {
      this->bounds.assign(bounds, bounds + 2 * dim);
      volume = calculate_volume();
      unif = new std::uniform_real_distribution<T>(0, 1);
    }

    ~UniformRectangularDomain() {
      delete unif;
    }

    inline int get_dim() const { return dim; }
    inline T get_volume() const { return volume; }
    inline T a(const int& i) const { return bounds[2 * i]; }
    inline T b(const int& i) const { return bounds[2 * i + 1]; }

    inline void sample(T* x) {
      for(int i=0; i<get_dim(); i++) {
        const T ai = a(i);
        const T bi = b(i);
        x[i] = (bi - ai) * (*unif)(rng) + ai;
      }
    }
};

} // namespace pift
#endif // PIFT_DOMAIN_HPP
