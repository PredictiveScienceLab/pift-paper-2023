# A simple makefile for the pift project
#
# Author:
# 	Ilias Bilionis
#
# Date:
# 	12/21/2022
#
# Path to YAML-CPP library. This is required for reading configuration fiels
YAMLCPP=/opt/homebrew/Cellar/yaml-cpp/0.7.0/
# Path to GLS library
GSL=/opt/homebrew/Cellar/gsl/2.7.1
# The compiler you wish to use
CXX=g++
# Compile options
CXXFLAGS=-O3 -ffast-math -std=c++20 -I./src -I./examples \
				 -I$(YAMLCPP)/include -I$(GSL)/include
# Link options
LDFLAGS=-L$(YAMLCPP)/lib -lyaml-cpp -L$(GSL)/lib -lgsl -lgslcblas -lm

# all: test_domain test_fields test_hamiltonian test_posterior \
# 		 test_free test_prior_exp test_post_exp\
# 		 example01 example02 example02b
#all: example01 example02 example02b example03 #example03c
#all: example01 example02a example02b example03a example03b
all: example03b

clean:
	$(RM) examples/*.o

test_constrained_mean: test_constrained_mean.o
	$(CXX) -o tests/test_constrained_mean  tests/test_constrained_mean.o $(LDFLAGS)

test_constrained_mean.o: tests/test_constrained_mean.cpp
	$(CXX) -c tests/test_constrained_mean.cpp $(CXXFLAGS) -I./tests -o tests/test_constrained_mean.o

test_cov: test_cov.o
	$(CXX) -o tests/test_cov  tests/test_cov.o $(LDFLAGS)

test_cov.o: tests/test_cov.cpp
	$(CXX) -c tests/test_cov.cpp $(CXXFLAGS) -I./tests -o tests/test_cov.o

example01: example01.o
	$(CXX) -o examples/example01 examples/example01.o $(LDFLAGS)

example01.o: examples/example01.cpp
	$(CXX) -c examples/example01.cpp $(CXXFLAGS) -o examples/example01.o

example02a: example02a.o
	$(CXX) -o examples/example02a examples/example02a.o $(LDFLAGS)

example02a.o: examples/example02a.cpp
	$(CXX) -c examples/example02a.cpp $(CXXFLAGS) -o examples/example02a.o

example02b: example02b.o
	$(CXX) -o examples/example02b examples/example02b.o $(LDFLAGS)

example02b.o: examples/example02b.cpp
	$(CXX) -c examples/example02b.cpp $(CXXFLAGS) -o examples/example02b.o

example03a: example03a.o
	$(CXX) -o examples/example03a examples/example03a.o $(LDFLAGS)

example03a.o: examples/example03a.cpp
	$(CXX) -c examples/example03a.cpp $(CXXFLAGS) -o examples/example03a.o

example03b: example03b.o
	$(CXX) -o examples/example03b examples/example03b.o $(LDFLAGS)

example03b.o: examples/example03b.cpp
	$(CXX) -c examples/example03b.cpp $(CXXFLAGS) -o examples/example03b.o

example03c: example03c.o
	$(CXX) -o examples/example03c examples/example03c.o $(LDFLAGS)

example03c.o: examples/example03c.cpp
	$(CXX) -c examples/example03c.cpp $(CXXFLAGS) -o examples/example03c.o
