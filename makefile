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
CXXFLAGS=-O3 -ffast-math -std=c++20 -I./src -I./examples \
				 -I$(YAMLCPP)/include
# Link options
LDFLAGS=-L$(YAMLCPP)/lib -lyaml-cpp

all: tests/test_domain tests/test_fields tests/test_hamiltonian \
		 examples/example01

clean:
	$(RM) tests/*.o examples/*.o

test_domain: test_domain.o
	$(CXX) -o tests/test_domain tests/test_domain.o

test_domain.o: tests/test_domain.cpp
	$(CXX) -c tests/test_domain.cpp $(CXXFLAGS) -o tests/test_domain.o

test_fields: test_fields.o
	$(CXX) -o tests/test_fields tests/test_fields.o

test_fields.o: tests/test_fields.cpp
	$(CXX) -c tests/test_fields.cpp $(CXXFLAGS)

test_hamiltonian: test_hamiltonian.o
	$(CXX) -o tests/test_hamiltonian tests/test_hamiltonian.o

test_hamiltonian.o: tests/test_hamiltonian.cpp
	$(CXX) -c tests/test_hamiltonian.cpp $(CXXFLAGS) -o tests/test_hamiltonian.o

example01: example01.o
	$(CXX) -o examples/example01 examples/example01.o $(LDFLAGS)

example01.o: example01.cpp
	$(CXX) -c examples/example01.cpp $(CXXFLAGS) -o examples01.o
