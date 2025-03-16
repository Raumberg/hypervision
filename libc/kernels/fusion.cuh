#ifndef RT_FUSION_H
#define RT_FUSION_H

#include <stdio.h> 

static void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s:%d\n",
                cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

#endif