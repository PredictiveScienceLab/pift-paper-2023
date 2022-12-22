// Template classes about posteriors.
//
// Author:
//  Ilias Bilionis
// 
// Date:
//  12/21/2022
//

#ifndef PIFT_POSTERIOR_HPP
#define PIFT_POSTERIOR_HPP

#include "hamiltonian.hpp"
#include "likelihood.hpp"

namespace pift {

// An unbiased estimator for minus log posterior
template<typename T, typename UEH, typename UEL>
class UEGradWPostAtFixedTheta {
protected:
  UEH& prior;
  UEL& likelihood;
  const int dim_w;
  T* tmp;
  std::uniform_int_distribution<int>* unif_int;

public:
  UEGradWPostAtFixedTheta(UEH& prior, UEL& likelihood) : 
    prior(prior), likelihood(likelihood),
    dim_w(likelihood.get_dim_w())
  {
    tmp = new T[dim_w];
  }

  ~UEGradWPostAtFixedTheta() {
    delete tmp;
  }

  inline UEH& get_prior() { return prior; }
  inline UEL& get_likelihood() { return likelihood; }

  inline T operator()(const T* w, T* out) {
    const T p = prior(w, out);
    const T l = likelihood(w, tmp);
    for(int i=0; i<dim_w; i++)
      out[i] += tmp[i];
    return p + l;
  }
}; // UEGradWPostAtFixedTheta

} // namespace pift
#endif // PIFT_POSTERIOR_HPP
