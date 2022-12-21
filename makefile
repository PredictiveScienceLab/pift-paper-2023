YAMLCPP=/opt/homebrew/Cellar/yaml-cpp/0.7.0/
CXX=g++
CXXFLAGS=-O3 -ffast-math -std=c++20 -I./src -I./examples \
				 -I$(YAMLCPP)/include
LDFLAGS=-L$(YAMLCPP)/lib -lyaml-cpp

all: test_domain test_fields test_hamiltonian example01

-include $(DEPENDS)

test_domain: test_domain.o
	$(CXX) -o test_domain test_domain.o

test_domain.o: tests/test_domain.cpp
	$(CXX) -c tests/test_domain.cpp $(CXXFLAGS)

test_fields: test_fields.o
	$(CXX) -o test_fields test_fields.o

test_fields.o: tests/test_fields.cpp
	$(CXX) -c tests/test_fields.cpp $(CXXFLAGS)

test_hamiltonian: test_hamiltonian.o
	$(CXX) -o test_hamiltonian test_hamiltonian.o

test_hamiltonian.o: tests/test_hamiltonian.cpp
	$(CXX) -c tests/test_hamiltonian.cpp $(CXXFLAGS)

example01: example01.o
	$(CXX) -o example01 example01.o $(LDFLAGS)

example01.o: examples/example01.cpp
	$(CXX) -c examples/example01.cpp $(CXXFLAGS)
