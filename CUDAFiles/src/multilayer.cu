#include "multilayer.h"
#include "cuda.h"
#include "cufft.h"
#include "kernels.h"
#include <vector>
#include <iostream>
#include <math.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include "stdio.h"
#include "cudaDebug.h"
#include "blur.h"

MultiLayer::MultiLayer(int width, int height, std::vector<double> z, double dx, double lambda, double n) :width(width), height(height)
{
    numLayers = (int)z.size();
    count = width*height;
    m_count = width*height*numLayers;
    blur = new Blur();

    allocate();
    multilayerPropagator(z, dx, lambda, n);
    conjugate<<<N_BLOCKS,N_THREADS>>>(m_count, Hq, Hn);
}

void MultiLayer::allocate(){
    cudaMalloc(&model, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&Hq, m_count*sizeof(cufftComplex));
    cudaMalloc(&Hn, m_count*sizeof(cufftComplex));
    cudaMalloc(&propagation, m_count*sizeof(cufftComplex));
    cudaMalloc(&guess, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&newGuess, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&u, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&temporary, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&sumArr, 2*N_BLOCKS*sizeof(double));
    cudaMalloc(&c, sizeof(double));
    cudaMalloc(&image, count*sizeof(double));
    cudaMalloc(&Imodel, count*sizeof(double));
    cudaMalloc(&temporaryf, 2*m_count*sizeof(double));

    modulus = (double*)malloc(m_count*sizeof(double));
    phase = (double*)malloc(m_count*sizeof(double));

    int n[2] = {height, width};
    cufftPlanMany(&fftPlan, 2, n, NULL, 1, count, NULL, 1, count, CUFFT_C2C, 2);
}

void MultiLayer::multilayerPropagator(std::vector<double> z, double dx, double lambda, double n){
    cufftComplex *placeHolder;
    for(int i = 0; i < numLayers; i++){
        placeHolder = &Hq[i*count];
        propagator<<<N_BLOCKS,N_THREADS>>>(width, height, z[i], dx, n, lambda, placeHolder);
    }
}

void MultiLayer::propagate(cufftComplex* kernel, cufftDoubleComplex* input, cufftDoubleComplex* out){
    Z2C<<<N_BLOCKS,N_THREADS>>>(m_count, input, propagation);
    cufftExecC2C(fftPlan, propagation, propagation, CUFFT_FORWARD);
    multiply<<<N_BLOCKS, N_THREADS>>>(m_count, kernel, propagation);
    cufftExecC2C(fftPlan, propagation, propagation, CUFFT_INVERSE);
    C2Z<<<N_BLOCKS,N_THREADS>>>(m_count, propagation, out);
}

void MultiLayer::calculateCost(double mu, double* model, cufftDoubleComplex* guess, double* temp, double* out){
    absolute<<<N_BLOCKS,N_THREADS>>>(m_count, guess, &temp[m_count]);
    square<<<N_BLOCKS,N_THREADS>>>(count, model, &temp[count]);

    sum<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(double)>>>(m_count, &temp[m_count], sumArr);
    sum<<<1,N_BLOCKS,N_BLOCKS*sizeof(double)>>>(N_BLOCKS, sumArr, sumArr);
    sum<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(double)>>>(count, &temp[count], &sumArr[N_BLOCKS]);
    sum<<<1,N_BLOCKS,N_BLOCKS*sizeof(double)>>>(N_BLOCKS, &sumArr[N_BLOCKS], &sumArr[N_BLOCKS]);
    
    cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
    simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&out[0]);
}

void MultiLayer::iterate(double *input, int iters, double mu, double* rconstr, double* iconstr, bool b_cost){
    // Initialization of variables
    s = 1;
    cudaMalloc(&cost, (1+iters)*sizeof(double));
    if(b_cost){
        h_cost = (double*)malloc((iters+1)*sizeof(double));
    }

    //Copying the input image from host to device memory - computationally complex
    cudaMemcpy(image, input, count*sizeof(double), cudaMemcpyHostToDevice);
    blur->gaussianBlur(width,height, 5, 3, image, temporaryf, image);

    //Copying the device memory image to device memory guesses
    for(int i = 0; i < numLayers; i++){
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, &guess[i*count]);
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, &u[i*count]);
    }

    for(int iter = 0; iter < iters; iter++){
        //Calculating the current iteration model 
        propagate(Hq, u, temporary);

        //Calculation of Imodel and model arrays
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, temporary, model, Imodel);

        //Calculation of the optimal scaling parameter c
        sumOfProducts<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(double)>>>(count, image, Imodel, sumArr);
        sum<<<1,N_BLOCKS,N_BLOCKS*sizeof(double)>>>(N_BLOCKS, sumArr, sumArr);
        sumOfProducts<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(double)>>>(count, Imodel, Imodel, &sumArr[N_BLOCKS]);
        sum<<<1,N_BLOCKS,N_BLOCKS*sizeof(double)>>>(N_BLOCKS, &sumArr[N_BLOCKS], &sumArr[N_BLOCKS]);

        simpleDivision<<<1,1>>>(sumArr, &sumArr[N_BLOCKS], c);

        //Cost calculation with sparsity constraint
        linear<<<N_BLOCKS,N_THREADS>>>(count, c, image, Imodel, temporaryf, false);

        if(b_cost)
            calculateCost(mu, temporaryf, guess, temporaryf, &cost[iter]);
        
        //Calculating residues
        multiplyfc<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, model);
        extend<<<N_BLOCKS,N_THREADS>>>(count, numLayers, model, temporary);
        propagate(Hn, temporary, temporary);

        cMultiplyfcp<<<N_BLOCKS,N_THREADS>>>(m_count, c, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, u, temporary, newGuess, false);

        //Applying strict bounds
        for(int i = 0 ; i < numLayers ; i++){
            strictBounds<<<N_BLOCKS,N_THREADS>>>(count, &newGuess[count*i], rconstr[i*2], rconstr[i*2+1], iconstr[i*2], iconstr[i*2+1]);
        }

        //Applying soft thresholding bounds
        softBounds<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, mu, 0.5f);

        double s_new = 0.5*(1+std::sqrt(1+4*s*s));
        double temp = (s-1)/s_new;
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, guess, temporary, false);
        cMultiplyfc<<<N_BLOCKS,N_THREADS>>>(m_count, temp, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, temporary, u, true);

        s = s_new;
        cudaMemcpy(guess, newGuess, m_count*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
    
    }
    /*
    // Final cost calculation
    //Calculating the current iteration model
    for(int i = 0; i < numLayers; i++){
        placeHolder = &temporary[i*width*height];
        HplaceHolder = &Hq[i*width*height];
        propagate(HplaceHolder, &u[i*count], placeHolder);
    }
    modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, temporary, model, Imodel);

    multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, imagef, temporaryf);
    multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, Imodel, &temporaryf[count]);

    sum<<<1, N_THREADS, N_THREADS*sizeof(double)>>>(count, temporaryf, sumArr);
    sum<<<1, N_THREADS, N_THREADS*sizeof(double)>>>(count, &temporaryf[count], &sumArr[1]);

    simpleDivision<<<1,1>>>(sumArr, &sumArr[1], c);

    linear<<<N_BLOCKS,N_THREADS>>>(count, c, imagef, Imodel, temporaryf, false);

    absolute<<<N_BLOCKS,N_THREADS>>>(m_count, guess, &temporaryf[2*count]);
    square<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, &temporaryf[count]);

    sum<<<1,N_THREADS, N_THREADS*sizeof(double)>>>(m_count, &temporaryf[2*count], sumArr);
    sum<<<1, N_THREADS, N_THREADS*sizeof(double)>>>(count, &temporaryf[count], &sumArr[1]);

    //Cost calculation with sparsity constraint
    cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
    simpleSum<<<1,1>>>(&sumArr[1],sumArr,&cost[iters]);
    */
    // Moving results to host memory
    offset<<<N_BLOCKS,N_THREADS>>>(m_count, 1.0f, 0.0f, guess, temporary);
    absolute<<<N_BLOCKS,N_THREADS>>>(m_count,temporary,temporaryf);
    for(int i = 0; i < numLayers; i++){
        maximum<<<1,N_THREADS,N_THREADS*sizeof(double)>>>(count, &temporaryf[i*count], sumArr);
        cDividefp<<<N_BLOCKS,N_THREADS>>>(count, sumArr, &temporaryf[i*count], &temporaryf[i*count]);
    }
    offsetf<<<N_BLOCKS,N_THREADS>>>(count, 1.f, &temporaryf[count], &temporaryf[count],false);
    for(int i = 0; i < numLayers; i++){
        maximum<<<1,N_THREADS,N_THREADS*sizeof(double)>>>(count, &temporaryf[i*count], sumArr);
        cDividefp<<<N_BLOCKS,N_THREADS>>>(count, sumArr, &temporaryf[i*count], &temporaryf[i*count]);
    }
    gpuErrchk(cudaMemcpy(modulus,temporaryf,m_count*sizeof(double),cudaMemcpyDeviceToHost));
    angle<<<N_BLOCKS,N_THREADS>>>(m_count,temporary,temporaryf);
    gpuErrchk(cudaMemcpy(phase,temporaryf,m_count*sizeof(double),cudaMemcpyDeviceToHost));
    if(b_cost){
        gpuErrchk(cudaMemcpy(h_cost, cost, (iters+1)*sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(cost);
    }
    gpuErrchk(cudaPeekAtLastError());

}

MultiLayer::~MultiLayer(){
    cudaFree(Hq);
    cudaFree(Hn);
    cudaFree(temporary);
    cudaFree(image);
    cudaFree(model);
    cudaFree(guess);
    cudaFree(newGuess);
    cudaFree(u);
    cudaFree(temporaryf);
    cudaFree(c);
    cudaFree(propagation);
    cufftDestroy(fftPlan);
    free(modulus);
    free(phase);
}