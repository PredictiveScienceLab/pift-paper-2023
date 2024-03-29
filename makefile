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

all: example01 example02a example02b example03a example03b

clean:
	$(RM) examples/*.o

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
