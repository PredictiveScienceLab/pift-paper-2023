CXX=g++
CXXFLAGS=-O3 -ffast-math -std=c++17 -I./src -I./examples

all: test_domain test_fields test_hamiltonian

-include $(DEPENDS)

test_domain: test_domain.o
	$(CXX) -o test_domain test_domain.o  $(CXXFLAGS)

test_domain.o: tests/test_domain.cpp
	$(CXX) -c tests/test_domain.cpp $(CXXFLAGS)

test_fields: test_fields.o
	$(CXX) -o test_fields test_fields.o $(CXXFLAGS)

test_fields.o: tests/test_fields.cpp
	$(CXX) -c tests/test_fields.cpp $(CXXFLAGS)

test_hamiltonian: test_hamiltonian.o
	$(CXX) -o test_hamiltonian test_hamiltonian.o $(CXXFLAGS)

test_hamiltonian.o: tests/test_hamiltonian.cpp
	$(CXX) -c tests/test_hamiltonian.cpp $(CXXFLAGS)
