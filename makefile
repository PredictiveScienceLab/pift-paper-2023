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
# The compiler you wish to use
CXX=g++
# Compile options
#CXXFLAGS=-O3 -ffast-math -std=c++20 -I./src -I./examples \
#				 -I$(YAMLCPP)/include
CXXFLAGS=-g -std=c++20 -I./src -I./examples \
				 -I$(YAMLCPP)/include
# Link options
LDFLAGS=-L$(YAMLCPP)/lib -lyaml-cpp

# all: test_domain test_fields test_hamiltonian test_posterior \
# 		 test_free test_prior_exp test_post_exp\
# 		 example01 example02 example02b
all: example03

clean:
	$(RM) tests/*.o examples/*.o

test_domain: test_domain.o
	$(CXX) -o tests/test_domain tests/test_domain.o

test_domain.o: tests/test_domain.cpp
	$(CXX) -c tests/test_domain.cpp $(CXXFLAGS) -o tests/test_domain.o

test_fields: test_fields.o
	$(CXX) -o tests/test_fields tests/test_fields.o

test_fields.o: tests/test_fields.cpp
	$(CXX) -c tests/test_fields.cpp $(CXXFLAGS) -o tests/test_fields.o

test_hamiltonian: test_hamiltonian.o
	$(CXX) -o tests/test_hamiltonian tests/test_hamiltonian.o

test_hamiltonian.o: tests/test_hamiltonian.cpp
	$(CXX) -c tests/test_hamiltonian.cpp $(CXXFLAGS) -o tests/test_hamiltonian.o

test_posterior: test_posterior.o
	$(CXX) -o tests/test_posterior tests/test_posterior.o

test_posterior.o: tests/test_posterior.cpp
	$(CXX) -c tests/test_posterior.cpp $(CXXFLAGS) -o tests/test_posterior.o

test_free: test_free.o
	$(CXX) -o tests/test_free tests/test_free.o

test_free.o: tests/test_free.cpp
	$(CXX) -c tests/test_free.cpp $(CXXFLAGS) -o tests/test_free.o

test_prior_exp: test_prior_exp.o
	$(CXX) -o tests/test_prior_exp tests/test_prior_exp.o $(LDFLAGS)

test_prior_exp.o: tests/test_prior_exp.cpp
	$(CXX) -c tests/test_prior_exp.cpp $(CXXFLAGS) -o tests/test_prior_exp.o

test_post_exp: test_post_exp.o
	$(CXX) -o tests/test_post_exp tests/test_post_exp.o $(LDFLAGS)

test_post_exp.o: tests/test_post_exp.cpp
	$(CXX) -c tests/test_post_exp.cpp $(CXXFLAGS) -o tests/test_post_exp.o

test_constructor: test_constructor.o
	$(CXX) -o tests/test_constructor tests/test_constructor.o $(LDFLAGS)

test_constructor.o: tests/test_constructor.cpp
	$(CXX) -c tests/test_constructor.cpp $(CXXFLAGS) -o tests/test_constructor.o

example01: example01.o
	$(CXX) -o examples/example01 examples/example01.o $(LDFLAGS)

example01.o: examples/example01.cpp
	$(CXX) -c examples/example01.cpp $(CXXFLAGS) -o examples/example01.o

example02: example02.o
	$(CXX) -o examples/example02 examples/example02.o $(LDFLAGS)

example02.o: examples/example02.cpp
	$(CXX) -c examples/example02.cpp $(CXXFLAGS) -o examples/example02.o

example02b: example02b.o
	$(CXX) -o examples/example02b examples/example02b.o $(LDFLAGS)

example02b.o: examples/example02b.cpp
	$(CXX) -c examples/example02b.cpp $(CXXFLAGS) -o examples/example02b.o

example03: example03.o
	$(CXX) -o examples/example03 examples/example03.o $(LDFLAGS)

example03.o: examples/example03.cpp
	$(CXX) -c examples/example03.cpp $(CXXFLAGS) -o examples/example03.o
