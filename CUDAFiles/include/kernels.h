#ifndef KERNELS_H
#define KERNELS_H

#include "cuda.h"
#include "cufft.h"

#define SQUARE(x) x*x
#define N_BLOCKS 16
#define N_THREADS 128

__global__ void propagator(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq);
__global__ void multiply(int count, cufftComplex* in, cufftComplex* out);
__global__ void multiplyfc(int count, float* in, cufftComplex* out);
__global__ void multiplyf(int count, float* in1, float* in2, float* out);
__global__ void absolute(int count, cufftComplex* in, float* out);
__global__ void angle(int count, cufftComplex* in, float* out);
__device__ void warpReduce(volatile float *sdata, int thIdx);
__global__ void sum(int count, float* in, float* result);
__global__ void maximum(int count, float* in, float* result);
__global__ void F2C(int count, float* in, cufftComplex* out);
__global__ void modelFunc(int count, int numLayers, float rOffset, float iOffset, cufftComplex* in, cufftComplex* out);
__global__ void ImodelFunc(int count, cufftComplex* in, float* out);
__global__ void conjugate(int count, cufftComplex *in, cufftComplex* out);
__global__ void simpleDivision(float* num, float* div, float* res);
__global__ void linear(int count, float* coef, float* constant, float* in, float* out, bool sign);
__global__ void square(int count, float* in, float* out);
__global__ void simpleSum(float* in1, float* in2, float* out);
__global__ void cMultiplyf(int count, float constant, float* in, float* out);
__global__ void cMultiply(int count, cufftComplex* constant, cufftComplex* in, cufftComplex* out);
__global__ void cMultiplyfc(int count, float constant, cufftComplex* in, cufftComplex* out);
__global__ void cMultiplyfcp(int count, float *constant, cufftComplex* in, cufftComplex* out);
__global__ void cDividefp(int count, float *constant, float* in, float* out);
__global__ void add(int count, cufftComplex* in1, cufftComplex* in2, cufftComplex* out, bool sign);
__global__ void strictBounds(int count, cufftComplex* arr, float r_min, float r_max, float i_min, float i_max);
__global__ void softBounds(int count, cufftComplex* arr, float mu, float t);

#endif