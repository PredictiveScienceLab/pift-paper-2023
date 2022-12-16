/* Some functions related to input output.
 *
 * Author:
 *  Ilias Bilionis
 *
 * Date:
 * 12/16/2022
 *
 */

#ifndef PIFT_IO_HPP
#define PIFT_IO_HPP

#include <iostream>
#include <string>
#include <fstream>


using namespace std;


template <typename T, typename S>
void cout_vec(
  const T* v,
  const int n,
  S& out,
  const string& prefix=""
) {
  out << prefix;
  for(int i=0; i<n; i++)
    out << v[i] << " ";
  out << endl;
}

template <typename T>
inline void cout_vec(const T* v, const int n, const string& prefix="") {    
  cout_vec(v, n, cout, prefix);
}

template <typename T, typename S>
inline void cout_vec(const vector<T>& x, S& out, const string& prefix="") {
  cout_vec(x.data(), x.size(), out, prefix);
}

template <typename T>
inline void cout_vec(const vector<T>& x, const string& prefix) {
  cout_vec(x, cout, prefix);
}

// Saves a vector to a file
template <typename T>
void savetxt(const T* x, const int& N, const string& filename) {
  ofstream of;
  of.open(filename);
  for(int i=0; i<N; i++)
    of << x[i] << " ";
  of << endl;
  of.close();
}

// Saves a matrix to a file
template <typename T>
void savetxt(const T* x, const int& N, const int& M, const string& filename) {
  ofstream of(filename);
  for(int i=0; i<N; i++) {
    for(int j=0; j<M; j++)
      of << x[i * M + j] << " ";
    of << endl;
  }
  of.close();
}

// Reads a vector from a file
template <typename T>
vector<T> loadtxtvec(const string& filename) {
  ifstream iff(filename);
  string line;
  vector<T> result;
  while(getline(iff, line)) {
    const T x = static_cast<T>(stod(line));
    result.push_back(x);
  }
  iff.close();
  return result;
}
#endif // PIFT_IO_HPP
