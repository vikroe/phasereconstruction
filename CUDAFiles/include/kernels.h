#ifndef KERNELS_H
#define KERNELS_H

#include "cuda.h"
#include "cufft.h"

#define SQUARE(x) x*x
#define N_BLOCKS 16
#define N_THREADS 128

__global__ void propagator(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq);
__global__ void multiply(int N, int M, cufftComplex* in, cufftComplex* out);
__global__ void multiplyf(int N, int M, float* in1, float* in2, float* out);
__device__ void warpReduce(volatile float *sdata, int thIdx);
__global__ void sum(int N, int M, float* in, float* result);
__global__ void F2C(int N, int M, float* in, cufftComplex* out);
__global__ void modelFunc(int N, int M, int count, float rOffset, float iOffset, cufftComplex* in, cufftComplex* out);
__global__ void ImodelFunc(int N, int M, cufftComplex* in, float* out);
__global__ void conjugate(int N, int M, cufftComplex *in, cufftComplex* out);
__global__ void simpleDivision(float* in, float* out);

#endif