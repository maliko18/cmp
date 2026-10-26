NVCC ?= nvcc
CXX ?= g++
CUDA_ARCH ?= sm_75
COMMON_FLAGS = -Icommon
CUDA_FLAGS = -O2 -std=c++17 -arch=$(CUDA_ARCH) $(COMMON_FLAGS)
SEQ_FLAGS = -O2 -std=c++17 $(COMMON_FLAGS)

.PHONY: all gpu seq check check-seq clean

all: pso_gpu pso_seq de_gpu de_seq

gpu: pso_gpu de_gpu

seq: pso_seq de_seq

pso_gpu: pso/gpu/main.cpp pso/gpu/kernel.cu pso/gpu/kernel.cpp pso/gpu/kernel.h common/benchmarks.h common/params.h
	$(NVCC) $(CUDA_FLAGS) -Ipso/gpu -o $@ pso/gpu/main.cpp pso/gpu/kernel.cu pso/gpu/kernel.cpp

pso_seq: pso/seq/pso_seq.cpp common/benchmarks.h common/params.h
	$(CXX) $(SEQ_FLAGS) -o $@ pso/seq/pso_seq.cpp

de_gpu: de/gpu/de_main.cpp de/gpu/de_kernel.cu de/gpu/de_kernel.h common/benchmarks.h common/params.h
	$(NVCC) $(CUDA_FLAGS) -Ide/gpu -o $@ de/gpu/de_main.cpp de/gpu/de_kernel.cu

de_seq: de/seq/de_seq.cpp common/benchmarks.h common/params.h
	$(CXX) $(SEQ_FLAGS) -o $@ de/seq/de_seq.cpp

check: check-seq

check-seq: pso_seq de_seq
	./pso_seq --check-benchmarks
	./de_seq --check-benchmarks

clean:
	rm -f pso_gpu pso_seq de_gpu de_seq
