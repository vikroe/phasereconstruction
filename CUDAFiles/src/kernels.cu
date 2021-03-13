#include "kernels.h"
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cufft.h"

__global__ void propagator(int N, int M, float z, float dx, float n, float lambda, cufftComplex* Hq){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float FX, FY, temp, res;
    float pre = n/lambda;
    float calc = 1/dx;
    int newIndex;
    int count = N*M;
    for (int i = index; i < count; i += stride)
    {
        newIndex = (i + count/2-1) % (count);
        FX = ((float)(1+(i/M)) * calc/(float)(N)) - calc/2.0f;
        FY = ((float)(1+(i%M)) * calc/(float)(M)) - calc/2.0f;
        res = 2 * M_PI*z*pre * sqrt(1 - SQUARE(FX/pre) - SQUARE(FY/pre));
        if(temp == 0.0){
            Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuComplex(0,0);
        }
        else{
            Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuComplex(std::cos(res),std::sin(res));
        }
    }
}

__global__ void multiply(int N, int M, cufftComplex*  in, cufftComplex*  out){
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        temp = make_cuFloatComplex(out[i].x/(float)(N*M), out[i].y/(float)(N*M));
        out[i] = cuCmulf(in[i], temp);
    }
}

__global__ void F2C(int N, int M, float*  in, cufftComplex*  out){
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        out[i] = make_cuFloatComplex(in[i], 0);
    }
}

__global__ void modelFunc(int N, int M, int count, float rOffset, float iOffset, cufftComplex* in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        out[i] = make_cuFloatComplex(rOffset, iOffset);
        for(int j = 0; j < count; j++){
            out[i] = cuCaddf(out[i], in[i + i*N*M]);
        }
    }
}

__global__ void ImodelFunc(int N, int M, cufftComplex* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        out[i] = SQUARE(cuCabsf(in[i]));
    }
}

__global__ void conjugate(int N, int M, cufftComplex *in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < N*M; i += stride){
        out[i] = cuConjf(in[i]);
    }
}