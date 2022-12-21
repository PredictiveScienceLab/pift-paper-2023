// Some postprocessing functions.
//
// Author:
//  Ilias Bilionis
//
// Date:
//  12/21/2022
//

#ifndef EXAMPLES_POST_PROCESSING_HPP
#define EXAMPLES_POST_PROCESSING_HPP

#include <string>

#include "pift.hpp"

template<typename T, typename FA, typename D>
inline void postprocess(
    const FA& phi, const D& domain, const int& n,
    std::string& samples_out_file, std::string& prefix
) {
  T x[n];
  pift::linspace(domain.a(0), domain.b(0), n, x);
  std::string x_file = prefix + "_x.csv";
  pift::savetxt(x, n, prefix + "_x.csv");
  auto ws = pift::loadtxtmat<T>(samples_out_file);  
  T phis[ws.size() * n];
  for(int i=0; i<ws.size(); i++)
    for(int j=0; j<n; j++)
      phis[n * i + j] = phi(x + j, ws[i].data() + 1);
  std::string phi_file = prefix + "_phi.csv";
  pift::savetxt(phis, ws.size(), n, phi_file);
}

#endif // EXAMPLES_POST_PROCESSING_HPP
