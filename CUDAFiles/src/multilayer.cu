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

    cudaMalloc(&Hq, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&Hn, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&res, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&guess, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&newGuess, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&u, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&temporary, m_count*sizeof(cufftDoubleComplex));
    cudaMalloc(&temporaryf, 2*m_count*sizeof(double));

    cudaMalloc(&image, count*sizeof(double));
    cudaMalloc(&sumArr, 2*sizeof(double));
    cudaMalloc(&c, sizeof(double));
    
    cudaMalloc(&model, count*sizeof(cufftDoubleComplex));
    cudaMalloc(&Imodel, count*sizeof(double));

    modulus = (double*)malloc(m_count*sizeof(double));
    phase = (double*)malloc(m_count*sizeof(double));

    cufftPlan2d(&fftPlan, width, height, CUFFT_Z2Z);

    multilayerPropagator(z, dx, lambda, n);
    conjugate<<<N_BLOCKS,N_THREADS>>>(m_count, Hq, Hn);

}

void MultiLayer::multilayerPropagator(std::vector<double> z, double dx, double lambda, double n){
    cufftDoubleComplex *placeHolder;
    for(int i = 0; i < numLayers; i++){
        placeHolder = &Hq[i*count];
        propagator<<<N_BLOCKS,N_THREADS>>>(width, height, z[i], dx, n, lambda, placeHolder);
    }
}

void MultiLayer::propagate(cufftDoubleComplex* kernel, cufftDoubleComplex* input, cufftDoubleComplex* out){
    cufftExecZ2Z(fftPlan, input, out, CUFFT_FORWARD);
    multiply<<<N_BLOCKS, N_THREADS>>>(count,kernel,out);
    cufftExecZ2Z(fftPlan, out, out, CUFFT_INVERSE);
}

void MultiLayer::iterate(double *input, int iters, double mu, double* rconstr, double* iconstr){
    // Initialization of variables
    s = 1;
    double t = 0.5;
    h_cost = (double*)malloc((iters+1)*sizeof(double));

    //Allocating the device memory array for cost at each iteration
    cudaMalloc(&cost, (1+iters)*sizeof(double));

    //Copying the input image from host to device memory - computationally complex
    cudaMemcpy(image, input, count*sizeof(double), cudaMemcpyHostToDevice);
    blur->gaussianBlur(width,height, 5, 2, image, temporaryf, image);

    //Copying the device memory image to device memory guesses
    for(int i = 0; i < numLayers; i++){
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, &guess[i*count]);
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, &u[i*count]);
    }

    for(int iter = 0; iter < iters; iter++){
        //Calculating the current iteration model
        for(int i = 0; i < numLayers; i++){
            propagate(&Hq[i*count], &u[i*count], &temporary[i*count]);
        }

        //Calculation of Imodel and model arrays
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, temporary, model, Imodel);

        //Calculation of the optimal scaling parameter c
        multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, image, temporaryf);
        multiplyf<<<N_BLOCKS, N_THREADS>>>(count, Imodel, Imodel, &temporaryf[count]);

        sum<<<1, N_THREADS, N_THREADS*sizeof(double)>>>(count, temporaryf, sumArr);
        sum<<<1, N_THREADS, N_THREADS*sizeof(double)>>>(count, &temporaryf[count], &sumArr[1]);

        simpleDivision<<<1,1>>>(sumArr, &sumArr[1], c);

        //Cost calculation with sparsity constraint
        linear<<<N_BLOCKS,N_THREADS>>>(count, c, image, Imodel, temporaryf, false);

        absolute<<<N_BLOCKS,N_THREADS>>>(m_count,guess,&temporaryf[m_count]);
        square<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, &temporaryf[count]);

        sum<<<1,N_THREADS, N_THREADS*sizeof(double)>>>(m_count,&temporaryf[m_count],sumArr);
        sum<<<1,N_THREADS, N_THREADS*sizeof(double)>>>(count, &temporaryf[count], &sumArr[1]);
        
        cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
        simpleSum<<<1,1>>>(&sumArr[1],sumArr,&cost[iter]);
        
        //Calculating residues
        multiplyfc<<<N_BLOCKS,N_THREADS>>>(m_count, temporaryf, temporary);
        for(int i = 0; i < numLayers; i++){
            propagate(&Hn[i*count], temporary, &res[i*count]);
        }

        cMultiplyf<<<1,1>>>(1,(2*t),c,c);
        cMultiplyfcp<<<N_BLOCKS,N_THREADS>>>(m_count, c, res, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, u, temporary, newGuess, false);

        //Applying strict bounds
        for(int i = 0 ; i < numLayers ; i++){
            strictBounds<<<N_BLOCKS,N_THREADS>>>(count, &newGuess[count*i], rconstr[i*2], rconstr[i*2+1], iconstr[i*2], iconstr[i*2+1]);
        }

        //Applying soft thresholding bounds
        //softBounds<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, mu, t);

        double s_new = 0.5*(1+std::sqrt(1+4*s*s));
        double temp = (s-1)/s_new;
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, guess, temporary, false);
        cMultiplyfc<<<N_BLOCKS,N_THREADS>>>(m_count, temp, temporary, res);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, res, u, true);

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
    gpuErrchk(cudaMemcpy(modulus,image,count*sizeof(double),cudaMemcpyDeviceToHost));
    angle<<<N_BLOCKS,N_THREADS>>>(m_count,temporary,temporaryf);
    gpuErrchk(cudaMemcpy(phase,temporaryf,m_count*sizeof(double),cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_cost, cost, (iters+1)*sizeof(double), cudaMemcpyDeviceToHost));
    // Deallocation of variables
    cudaFree(cost);
    gpuErrchk(cudaPeekAtLastError());

}

MultiLayer::~MultiLayer(){
    cudaFree(Hq);
    cudaFree(Hn);
    cudaFree(temporary);
    cudaFree(image);
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