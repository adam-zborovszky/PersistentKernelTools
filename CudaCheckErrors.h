#pragma once

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

#define cudaCheckErrors(ans) { CudaAssert((ans), __FILE__, __LINE__); }
inline __host__ __device__ void CudaAssert(cudaError_t code, const char* file, int line, bool abort = true) {
	if (code != cudaSuccess) {
#ifdef __CUDACC__
		printf("GPUassert: Error(%d) %s %s in line %d\n", (int)code, cudaGetErrorString(code), file, line);
		if (abort)
			assert(code);
#else
		fprintf(stderr, "GPUassert: Error(%d) %s %s in line %d\n", (int)code, cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
#endif
	}
}
