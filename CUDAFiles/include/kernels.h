#ifndef KERNELS_H
#define KERNELS_H

#include "cuda.h"
#include "cufft.h"

#define SQUARE(x) x*x
#define N_BLOCKS 16
#define N_THREADS 256

__global__ void propagator(int N, int M, double z, double dx, double n, double lambda, cufftComplex* Hq);
__global__ void multiply(int count, cufftComplex* in, cufftComplex* out);
__global__ void multiplyfc(int count, double* in, cufftDoubleComplex* out);
__global__ void multiplyf(int count, double* in1, double* in2, double* out);
__global__ void absolute(int count, cufftDoubleComplex* in, double* out);
__global__ void real(int count, cufftDoubleComplex* in, double* out);
__global__ void imag(int count, cufftDoubleComplex* in, double* out);
__global__ void angle(int count, cufftDoubleComplex* in, double* out);
__global__ void sum(int count, double* in, double* result);
__global__ void sumOfProducts(int count, double* in1, double* in2, double* result);
__global__ void maximum(int count, double* in, double* result);
__global__ void minimum(int count, double* in, double* result);
__global__ void F2C(int count, double* in, cufftDoubleComplex* out);
__global__ void modelFunc(int count, int numLayers, double rOffset, double iOffset, cufftDoubleComplex* in, cufftDoubleComplex* model, double* Imodel);
__global__ void conjugate(int count, cufftComplex *in, cufftComplex* out);
__global__ void simpleDivision(double* num, double* div, double* res);
__global__ void linear(int count, double* coef, double* constant, double* in, double* out, bool sign);
__global__ void square(int count, double* in, double* out);
__global__ void simpleSum(double* in1, double* in2, double* out);
__global__ void cMultiplyf(int count, double constant, double* in, double* out);
__global__ void cMultiply(int count, cufftDoubleComplex* constant, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void cMultiplyfc(int count, double constant, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void cMultiplyfcp(int count, double *constant, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void cDividefp(int count, double *constant, double* in, double* out);
__global__ void add(int count, cufftDoubleComplex* in1, cufftDoubleComplex* in2, cufftDoubleComplex* out, bool sign);
__global__ void strictBounds(int count, cufftDoubleComplex* arr, double r_min, double r_max, double i_min, double i_max);
__global__ void softBounds(int count, cufftDoubleComplex* arr, double mu, double t);
__global__ void rowConvolution(int N, int M, double diameter, double* kernel, double* image, double* output, bool horizontal);
__global__ void offset(int count, double roff, double ioff, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void offsetf(int count, double roff, double* in, double* out, bool sign);
__global__ void C2Z(int count, cufftComplex* in, cufftDoubleComplex* out);
__global__ void Z2C(int count, cufftDoubleComplex* in, cufftComplex* out);
__global__ void extend(int count, int multiple, cufftDoubleComplex* in, cufftDoubleComplex* out);
__global__ void D2u8(int count, double* in, uint8_t* out);

#endif