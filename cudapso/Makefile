NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= sm_75

.PHONY: all gpu seq check clean

all: pso_gpu pso_seq

gpu: pso_gpu

seq: pso_seq

pso_gpu: main.cpp kernel.cu kernel.h benchmarks.h
	$(NVCC) -O2 -arch=$(CUDA_ARCH) -o $@ main.cpp kernel.cu

pso_seq: pso_seq.cpp benchmarks.h
	$(CXX) -O2 -std=c++17 -o $@ pso_seq.cpp

check: all
	./pso_seq --check-benchmarks
	./pso_seq 1 10 50 3

clean:
	rm -f pso_gpu pso_seq
