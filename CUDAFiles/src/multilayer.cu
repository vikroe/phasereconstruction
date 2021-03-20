#include "multilayer.h"
#include "cuda.h"
#include "cufft.h"
#include "kernels.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <cuda_device_runtime_api.h>
#include <assert.h>
#include "stdio.h"
#include "string.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stdout,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void cudaMemoryTest()
{
    const unsigned int N = 1048576;
    const unsigned int bytes = N * sizeof(int);
    int *h_a = (int*)malloc(bytes);
    int *d_a;
    gpuErrchk(cudaMalloc((int**)&d_a, bytes));

    memset(h_a, 0, bytes);
    gpuErrchk(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost));
}

MultiLayer::MultiLayer(int width, int height, std::vector<float> z, float dx, float lambda, float n) :width{width}, height{height}
{
    numLayers = (int)z.size();
    count = width*height;
    m_count = width*height*numLayers;

    cudaMalloc(&Hq, m_count*sizeof(cufftComplex));
    cudaMalloc(&Hn, m_count*sizeof(cufftComplex));
    cudaMalloc(&res, m_count*sizeof(cufftComplex));
    cudaMalloc(&guess, m_count*sizeof(cufftComplex));
    cudaMalloc(&newGuess, m_count*sizeof(cufftComplex));
    cudaMalloc(&u, m_count*sizeof(cufftComplex));
    cudaMalloc(&temporary, m_count*sizeof(cufftComplex));
    cudaMalloc(&temporaryf, 2*m_count*sizeof(float));

    cudaMalloc(&image, count*sizeof(cufftComplex));
    cudaMalloc(&imagef, count*sizeof(float));
    cudaMalloc(&sumArr, 2*N_BLOCKS*sizeof(float));
    cudaMalloc(&c, sizeof(float));
    
    cudaMalloc(&model, count*sizeof(cufftComplex));
    cudaMalloc(&Imodel, count*sizeof(float));

    modulus = (float*)malloc(m_count*sizeof(float));
    phase = (float*)malloc(m_count*sizeof(float));

    cufftPlan2d(&fftPlan, width, height, CUFFT_C2C);

    multilayerPropagator(z, dx, lambda, n);

}

void MultiLayer::multilayerPropagator(std::vector<float> z, float dx, float lambda, float n){
    cufftComplex *placeHolder;
    for(int i = 0; i < numLayers; i++){
        placeHolder = &Hq[i*count];
        propagator<<<N_BLOCKS,N_THREADS>>>(width, height, z[i], dx, n, lambda, placeHolder);
        gpuErrchk(cudaPeekAtLastError());
    }
}

void MultiLayer::propagate(cufftComplex* kernel, cufftComplex* input, cufftComplex* out){
    cufftExecC2C(fftPlan, input, out, CUFFT_FORWARD);
    multiply<<<N_BLOCKS, N_THREADS>>>(count,kernel,out);
    cufftExecC2C(fftPlan, out, out, CUFFT_INVERSE);
    gpuErrchk(cudaPeekAtLastError());
}

void MultiLayer::iterate(float *input, int iters, float mu, float* rconstr, float* iconstr){
    // Initialization of variables
    s = 1;
    float t = 0.5;
    float *fplaceHolder;
    h_cost = (float*)malloc((iters+1)*sizeof(float));
    cufftComplex *placeHolder;
    cufftComplex *HplaceHolder;

    conjugate<<<N_BLOCKS,N_THREADS>>>(m_count, Hq, Hn);

    //Allocating the device memory array for cost at each iteration
    cudaMalloc(&cost, (1+iters)*sizeof(float));

    //Copying the input image from host to device memory - computationally complex
    cudaMemcpy(imagef, input, count*sizeof(float), cudaMemcpyHostToDevice);
    F2C<<<N_BLOCKS, N_THREADS>>>(count, imagef, image);

    //Copying the device memory image to device memory guesses
    for(int i = 0; i < numLayers; i++){
        cudaMemcpy(&guess[i*count], image, count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&u[i*count], image, count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    }
    gpuErrchk(cudaPeekAtLastError());

    for(int iter = 0; iter < iters; iter++){

        //Calculating the current iteration model
        for(int i = 0; i < numLayers; i++){
            placeHolder = &temporary[i*count];
            HplaceHolder = &Hq[i*count];
            propagate(HplaceHolder, image, placeHolder);
        }
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, temporary, model);
        ImodelFunc<<<N_BLOCKS,N_THREADS>>>(count, model, Imodel);

        fplaceHolder = temporaryf;
        multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, imagef, fplaceHolder);
        sum<<<1, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, sumArr);

        fplaceHolder = &temporaryf[count];
        multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, Imodel, fplaceHolder);
        sum<<<1, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, &sumArr[N_BLOCKS]);

        simpleDivision<<<1,1>>>(sumArr, &sumArr[N_BLOCKS], c);
        cudaMemoryTest();

        linear<<<N_BLOCKS,N_THREADS>>>(count, c, imagef, Imodel, temporaryf, false);

        absolute<<<N_BLOCKS,N_THREADS>>>(m_count,guess,&temporaryf[2*count]);
        sum<<<1,N_THREADS, N_THREADS*sizeof(float)>>>(m_count,&temporaryf[2*count],sumArr);

        square<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, &temporaryf[count]);
        sum<<<1,N_THREADS, N_THREADS*sizeof(float)>>>(m_count,&temporaryf[count],&sumArr[N_BLOCKS]);

        //Cost calculation with sparsity constraint
        cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
        simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&cost[iter]);
        
        multiplyfc<<<N_BLOCKS,N_THREADS>>>(m_count, temporaryf, temporary);
        for(int i = 0; i < numLayers; i++){
            placeHolder = &res[i*count];
            HplaceHolder = &Hn[i*count];
            propagate(HplaceHolder, temporary, placeHolder);
        }

        cMultiplyf<<<1,1>>>(1,(2*t),c,c);
        cMultiplyfcp<<<N_BLOCKS,N_THREADS>>>(m_count, c, res, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, u, temporary, newGuess, false);

        //Applying strict bounds
        for(int i = 0 ; i < numLayers ; i++){
            placeHolder = &newGuess[count*i];
            strictBounds<<<N_BLOCKS,N_THREADS>>>(count, placeHolder, rconstr[i*2], rconstr[i*2+1], iconstr[i*2], iconstr[i*2+1]);
        }

        //Applying soft thresholding bounds
        softBounds<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, mu, t);

        float s_new = 0.5*(1+std::sqrt(1+4*s*s));
        float temp = (s-1)/s_new;
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, guess, temporary, false);
        cMultiplyfc<<<N_BLOCKS,N_THREADS>>>(count, temp, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, temporary, u, false);

        s = s_new;
        cudaMemcpy(guess, newGuess, m_count*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    
    }

    // Final cost calculation
    //Calculating the current iteration model
    for(int i = 0; i < numLayers; i++){
        placeHolder = &temporary[i*width*height];
        HplaceHolder = &Hq[i*width*height];
        propagate(HplaceHolder, image, placeHolder);
    }
    modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, temporary, model);
    ImodelFunc<<<N_BLOCKS,N_THREADS>>>(count, model, Imodel);

    fplaceHolder = temporaryf;
    multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, imagef, fplaceHolder);
    sum<<<1, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, sumArr);
    fplaceHolder = &temporaryf[count];
    multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, Imodel, fplaceHolder);
    sum<<<1, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, &sumArr[N_BLOCKS]);
    gpuErrchk(cudaPeekAtLastError());

    simpleDivision<<<1,1>>>(sumArr, &sumArr[N_BLOCKS], c);

    linear<<<N_BLOCKS,N_THREADS>>>(count, c, imagef, Imodel, temporaryf, false);

    absolute<<<N_BLOCKS,N_THREADS>>>(m_count,guess,&temporaryf[2*count]);
    sum<<<1,N_THREADS, N_THREADS*sizeof(float)>>>(m_count,&temporaryf[2*count],sumArr);

    square<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, &temporaryf[count]);
    sum<<<1,N_THREADS, N_THREADS*sizeof(float)>>>(m_count,&temporaryf[count],&sumArr[N_BLOCKS]);

    //Cost calculation with sparsity constraint
    cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
    simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&cost[iters]);
    gpuErrchk(cudaPeekAtLastError());

    // Moving results to host memory

    absolute<<<N_BLOCKS,N_THREADS>>>(m_count,guess,temporaryf);
    maximum<<<1,N_THREADS,N_THREADS*sizeof(float)>>>(m_count, temporaryf, sumArr);
    cDividefp<<<N_BLOCKS,N_THREADS>>>(m_count,sumArr,temporaryf,temporaryf);
    gpuErrchk(cudaMemcpy(modulus,temporaryf,m_count*sizeof(float),cudaMemcpyDeviceToHost));
    angle<<<N_BLOCKS,N_THREADS>>>(m_count,guess,temporaryf);
    gpuErrchk(cudaMemcpy(phase,temporaryf,m_count*sizeof(float),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_cost, cost, iters+1, cudaMemcpyDeviceToHost));

    printf("Did we get here at all?\n");

    // Deallocation of variables
    cudaFree(cost);
    gpuErrchk(cudaPeekAtLastError());

}

MultiLayer::~MultiLayer(){
    cudaFree(Hq);
    cudaFree(Hn);
    cudaFree(temporary);
    cudaFree(image);
    cudaFree(imagef);
    cudaFree(res);
    cudaFree(model);
    cudaFree(guess);
    cudaFree(newGuess);
    cudaFree(u);
    cudaFree(temporaryf);
    cudaFree(c);
    cufftDestroy(fftPlan);
    free(h_cost);
    free(modulus);
    free(phase);
}