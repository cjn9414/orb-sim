# CPU make variables
CPU_CC=g++
ST=-DSTATIC_TREE
DIRECT=-DDIRECT
OF=-DONLY_FINAL
FLAGS=-Wall -std=c++11 $(ST)
CPU_SRC=common.cpp main.cpp
CPU_BIN=cpu_barnes

# GPU Make variables
GPU_CC=nvcc
GPU_SRC=common.cu
COMMON_GPU_OBJ=common.o
COMMON_GPU_SRC=common.cu
BH_GPU_OBJ=bh.o
BH_GPU_SRC=bh.cu
GPU_OBJ=$(COMMON_GPU_OBJ) $(BH_GPU_OBJ) 
CUDA_FLAGS=-lcudart -lcuda -L/opt/cuda-10.2/lib64
GPU_FLAGS=--std=c++11
GPU_BIN=gpu_barnes

.PHONY: help clean
.SILENT: help

help: 
	@echo "Make options: all, help, clean, $(CPU_BIN), $(GPU_BIN)"

all: $(CPU_BIN) $(GPU_BIN) 
	@echo "Build complete"

$(CPU_BIN):
	-@ln -s common.cu common.cpp || true 
	$(CPU_CC) -o $@ $(CPU_SRC) $(FLAGS)
	@echo "CPU build complete"

$(GPU_BIN):
	$(GPU_CC) -dc -o $(COMMON_GPU_OBJ) $(COMMON_GPU_SRC) $(GPU_FLAGS) 
	$(GPU_CC) -dc -o $(BH_GPU_OBJ) $(BH_GPU_SRC) $(GPU_FLAGS) 
	$(GPU_CC) -o $@ $(GPU_OBJ) $(CUDA_FLAGS) $(GPU_FLAGS)
	@echo "GPU build complete"

clean:
	-rm -rf $(CPU_BIN) $(GPU_BIN) 
	-rm -rf $(BH_GPU_OBJ) $(COMMON_GPU_OBJ)
