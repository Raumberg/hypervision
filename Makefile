# CUDA Compiler and Flags
NVCC = nvcc
CUDA_PATH = /usr/local/cuda
CFLAGS = -arch=sm_86 \         # GPU's compute capability
         --use_fast_math \     # Fast math optimizations
         -Xcompiler -fPIC \    # Position Independent Code
         -O3 \                 # Aggressive optimization
         --compiler-options="-Wall"

# Source and Output
SRC = rtFusion.cu
LIB_NAME = librtFusion.so
OBJ = rtFusion.o

# Default compute capability
ARCH ?= sm_86

all: build

build: $(LIB_NAME)

$(LIB_NAME): $(SRC)
	@echo "Compiling CUDA kernel with compute capability $(ARCH)..."
	$(NVCC) $(CFLAGS) -arch=$(ARCH) -shared -o $@ $<

clean:
	@echo "Cleaning build artifacts..."
	rm -f $(LIB_NAME) $(OBJ)

.PHONY: all build clean