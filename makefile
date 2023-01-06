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

# all: test_domain test_fields test_hamiltonian test_posterior \
# 		 test_free test_prior_exp test_post_exp\
# 		 example01 example02 example02b
#all: example01 example02 example02b example03 #example03c
all: example01

clean:
	$(RM) examples/*.o

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

example03b: example03b.o
	$(CXX) -o examples/example03b examples/example03b.o $(LDFLAGS)

example03b.o: examples/example03b.cpp
	$(CXX) -c examples/example03b.cpp $(CXXFLAGS) -o examples/example03b.o

example03c: example03c.o
	$(CXX) -o examples/example03c examples/example03c.o $(LDFLAGS)

example03c.o: examples/example03c.cpp
	$(CXX) -c examples/example03c.cpp $(CXXFLAGS) -o examples/example03c.o
