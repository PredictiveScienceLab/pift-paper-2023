/* Template class that represents the prolongation of a function.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/16/2022
 *
 */

#ifndef PIFT_PROLONGATION_HPP
#define PIFT_PROLONGATION_HPP

template <typename T>
class Prolongation {
private:
  int num_terms;

public:
  const int dim;
  const int degree;
  vector<T> data;

  Prolongation(const int dim, const int degree) :
    dim(dim), degree(degree)
  {
    if (dim == 1)
      num_terms = 1 + degree;
    else
      num_terms = (1 - pow(dim, degree + 1)) / (1 - dim); 
    data.resize(num_terms);
  }
};

#endif // PIFT_PROLONGATION
