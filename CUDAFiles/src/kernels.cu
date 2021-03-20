#include "kernels.h"
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cufft.h"

//This include is completely unnecessary and can be omitted - only used to prevent Intellisense from thinking CUDA variables are undefined
#include <device_launch_parameters.h>

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
        res = 2 * M_PI*z*pre * sqrtf(1 - SQUARE(FX/pre) - SQUARE(FY/pre));
        Hq[(newIndex % M) > M/2-1 ? newIndex-M/2 : newIndex+M/2] = make_cuComplex(cosf(res),sinf(res));
    }
}

__global__ void multiply(int count, cufftComplex*  in, cufftComplex* out){
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        temp = make_cuFloatComplex(in[i].x/(float)(count), in[i].y/(float)(count));
        out[i] = cuCmulf(out[i], temp);
    }
}

__global__ void multiplyf(int count, float*  in1, float*  in2, float*  out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in1[i]*in2[i];
    }
}

__global__ void multiplyfc(int count, float* in, cufftComplex* out){
    cufftComplex temp;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        temp = make_cuFloatComplex(in[i], 0);
        out[i] = cuCmulf(temp,out[i]);
    }
}

__global__ void absolute(int count, cufftComplex* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCabsf(in[i]);
    }
}

__global__ void angle(int count, cufftComplex* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = atan2f(in[i].y,in[i].x);
    }
}


__device__ void warpReduce(volatile float *sdata, int thIdx){
    if(N_THREADS>=64) sdata[thIdx] += sdata[thIdx + 32];
    if(N_THREADS>=32) sdata[thIdx] += sdata[thIdx + 16];
    if(N_THREADS>=16) sdata[thIdx] += sdata[thIdx + 8];
    if(N_THREADS>=8) sdata[thIdx] += sdata[thIdx + 4];
    if(N_THREADS>=4) sdata[thIdx] += sdata[thIdx + 2];
    if(N_THREADS>=2) sdata[thIdx] += sdata[thIdx + 1];
}

//Fast parallel sum 
__global__ void sum(int count, float* in, float* result){
    extern __shared__ float sharedIn[];
    int thIdx = threadIdx.x;
    int index = blockIdx.x*(N_THREADS*2) + thIdx;
    int stride = N_THREADS*2*gridDim.x;
    sharedIn[thIdx] = 0;
    
    while(index < count){
        sharedIn[thIdx] += in[index] + in[index+N_THREADS];
        index += stride;
    }
    __syncthreads();

    if (N_THREADS >= 512){
        if (thIdx < 256){
            sharedIn[thIdx] += sharedIn[thIdx + 256]; 
        } 
        __syncthreads();
    }
    if (N_THREADS >= 256){
        if (thIdx < 128){
            sharedIn[thIdx] += sharedIn[thIdx + 128];
        }
        __syncthreads();
    }
    if (N_THREADS >= 128){
        if (thIdx <  64){
            sharedIn[thIdx] += sharedIn[thIdx +  64];
        }
        __syncthreads();
    }
    if (thIdx < 32) warpReduce(sharedIn, thIdx);
    if (thIdx == 0) result[blockIdx.x] = sharedIn[0];
}

__global__ void maximum(int count, float* in, float* result){
    extern __shared__ float sharedIn[];
    int thIdx = threadIdx.x;
    int index = blockIdx.x*(N_THREADS*2) + thIdx;
    int stride = N_THREADS*2*gridDim.x;
    sharedIn[thIdx] = 0;
    
    while(index < count){
        sharedIn[thIdx] = (in[index] > in[index+N_THREADS]) ? in[index] : in[index+N_THREADS];
        index += stride;
    }
    __syncthreads();

    for(unsigned int i = blockDim.x/2 ; i>0 ; i>>=1){
        if(thIdx < i){
            sharedIn[thIdx] = (sharedIn[thIdx] > sharedIn[thIdx+i]) ? sharedIn[thIdx] : sharedIn[thIdx+i];
        }
        __syncthreads();
    }
    if (thIdx == 0) result[blockIdx.x] = sharedIn[0];
}


__global__ void F2C(int count, float*  in, cufftComplex*  out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = make_cuFloatComplex(in[i], 0);
    }
}

__global__ void modelFunc(int count, int numLayers, float rOffset, float iOffset, cufftComplex* in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = make_cuFloatComplex(rOffset, iOffset);
        for(int j = 0; j < numLayers; j++){
            out[i] = cuCaddf(out[i], in[i + j*count]);
        }
    }
}

__global__ void ImodelFunc(int count, cufftComplex* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = SQUARE(cuCabsf(in[i]));
    }
}

__global__ void conjugate(int count, cufftComplex *in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuConjf(in[i]);
    }
}

__global__ void simpleDivision(float* num, float* div, float* res){
        if(div[0] == 0.0f)
            div[0] = div[0] + 0.00001f;
        res[0] = num[0] / div[0];
}

__global__ void linear(int count, float* coef, float* constant, float* in, float* out, bool sign){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if(sign)
            out[i] = coef[0]*in[i] + constant[i];
        else
            out[i] = coef[0]*in[i] - constant[i];
    }
}

__global__ void square(int count, float* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = SQUARE(in[i]);
    }
}

__global__ void simpleSum(float* in1, float* in2, float* out){
    out[0] = in1[0] + in2[0];
}

__global__ void cMultiplyf(int count, float constant, float* in, float* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = constant*in[i];
    }
}

__global__ void cMultiply(int count, cufftComplex* constant, cufftComplex* in, cufftComplex* out){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCmulf(constant[0],in[i]);
    }
}

__global__ void cMultiplyfc(int count, float constant, cufftComplex* in, cufftComplex* out){
    cufftComplex temp = make_cuFloatComplex(constant, 0);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCmulf(temp,in[i]);
    }
}

__global__ void cMultiplyfcp(int count, float *constant, cufftComplex* in, cufftComplex* out){
    cufftComplex temp = make_cuFloatComplex(constant[0], 0);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = cuCmulf(temp,in[i]);
    }
}

__global__ void cDividefp(int count, float *constant, float* in, float* out){
    if(constant[0] == 0.0f){
        constant[0] += 0.00001;
    }
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        out[i] = in[i] / constant[0];
    }
}

__global__ void add(int count, cufftComplex* in1, cufftComplex* in2, cufftComplex* out, bool sign){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if (sign)
            out[i] = cuCaddf(in1[i], in2[i]);
        else
            out[i] = cuCsubf(in1[i], in2[i]); 
    }
}

__global__ void strictBounds(int count, cufftComplex* arr, float r_min, float r_max, float i_min, float i_max){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        if (arr[i].x < r_min)
            arr[i].x = r_min;
        else if (arr[i].x > r_max)
            arr[i].x = r_max;
        if (arr[i].y < i_min)
            arr[i].y = i_min;
        else if (arr[i].y > i_max)
            arr[i].y = i_max; 
    }
}

__global__ void softBounds(int count, cufftComplex* arr, float mu, float t){
    cufftComplex zero = make_cuFloatComplex(0,0);
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < count; i += stride){
        cufftComplex temp = make_cuFloatComplex(arr[i].x-mu*t,arr[i].y);
        if(cuCabsf(temp) < 0)
            arr[i] = zero;
        else
            arr[i] = temp;
    }
}
