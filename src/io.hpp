// Some functions related to input output.
//
// Author:
//  Ilias Bilionis
//
// Date:
// 12/16/2022

#ifndef PIFT_IO_HPP
#define PIFT_IO_HPP

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

namespace pift {

// Send a vectot to a stream
template<typename T, typename S>
void cout_vec(
  const T* v,
  const int n,
  S& out,
  const std::string& prefix=""
) {
  out << prefix;
  for(int i=0; i<n; i++)
    out << v[i] << " ";
  out << std::endl;
}

template<typename T>
inline void cout_vec(
    const T* v, const int n, const std::string& prefix=""
) {    
  cout_vec(v, n, std::cout, prefix);
}

template<typename T, typename S>
inline void cout_vec(
    const std::vector<T>& x, S& out, const std::string& prefix=""
) {
  cout_vec(x.data(), x.size(), out, prefix);
}

template<typename T>
inline void cout_vec(const std::vector<T>& x, const std::string& prefix) {
  cout_vec(x, std::cout, prefix);
}

// Saves a std::vector to a file
template<typename T>
void savetxt(
    const T* x, const int& n, const std::string& filename,
    const bool& disp=true
) {
  if(disp) {
    std::cout << "> writing " << n << " vector on " << filename;
    std::cout << std::endl;
  }
  std::ofstream of;
  of.open(filename);
  for(int i=0; i<n; i++)
    of << x[i] << " ";
  of << std::endl;
  of.close();
}

// Saves a matrix to a file
template<typename T>
void savetxt(
    const T* x, const int& n, const int& m, const std::string& filename,
    const bool& disp=true
) {
  if(disp) {
    std::cout << "> writing " << n << "x" << m << " matrix on " << filename;
    std::cout << std::endl;
  }
  std::ofstream of(filename);
  for(int i=0; i<n; i++) {
    for(int j=0; j<m; j++)
      of << x[i * m + j] << " ";
    of << std::endl;
  }
  of.close();
}

// Reads a std::vector from a file
template<typename T>
std::vector<T> loadtxtvec(const std::string& filename) {
  std::ifstream iff(filename);
  std::string line;
  std::vector<T> result;
  while(getline(iff, line)) {
    const T x = static_cast<T>(std::stod(line));
    result.push_back(x);
  }
  iff.close();
  return result;
}

// Reads a matrix from a file
template<typename T>
std::vector<std::vector<T>> loadtxtmat(
    const std::string& filename,
    const char& delimiter=' '
) {
  std::ifstream iff(filename);
  std::string line;
  std::vector<std::vector<T>> result;
  while(getline(iff, line)) {
    std::vector<T> row;
    std::stringstream s(line);
    std::string value;
    while(getline(s, value, delimiter)) {
      const T x = static_cast<T>(std::stod(value));
      row.push_back(x);
    }
    result.push_back(row);
  }
  return result;
}

} // namespace pift
#endif // PIFT_IO_HPP
