/* Some unit tests for the UniformRectangularDomain template class.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 *  12/19/2022
 *
 */

#include "domain.hpp"
#include "io.hpp"

#include <iomanip>

using namespace std;

int main(int argc, char* argv[]) {
  mt19937 rng;

  cout.precision(3);

  // 1D interval
  cout << "1D domain test" << endl;
  float bounds[2] = {-1.0, 3.0};
  UniformRectangularDomain<float, mt19937> Omega(bounds, 1, rng);
  float x[1];
  for(int i=0; i<10; i++) {
    Omega.sample(x);
    cout << i << " " << x[0] << endl;
  }

  // 2D interval
  cout << endl << "2D domain test" << endl;
  float bounds2[4] = {-1.0, 3.0, -5.0, 6.0};
  UniformRectangularDomain<float, mt19937> Omega2(bounds2, 2, rng); 
  float y[2];
  for(int i=0; i<10; i++) {
    Omega2.sample(y);
    cout << i << " (" << y[0] << ", " << y[1] << ")" << endl;
  }

  return 0;
}
