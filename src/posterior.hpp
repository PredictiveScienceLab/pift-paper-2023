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

public:
  UEGradWPostAtFixedTheta(UEH& prior, UEL& likelihood) : 
    prior(prior), likelihood(likelihood)
  {}

  inline UEH& get_prior() { return prior; }
  inline UEL& get_likelihood() { return likelihood; }

  inline T operator()(const T* w, T* out) {
    return prior(w, out) + likelihood(w, out);
  }
}; // UEGradWPostAtFixedTheta

} // namespace pift
#endif // PIFT_POSTERIOR_HPP
