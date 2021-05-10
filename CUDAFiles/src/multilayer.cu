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

MultiLayer::MultiLayer(int width, 
                        int height, 
                        std::vector<double> z,
                        std::vector<double> rconstr, 
                        std::vector<double> iconstr, 
                        double mu, 
                        double dx, 
                        double lambda, 
                        double n) :width(width), height(height), z(z), rconstr(rconstr), iconstr(iconstr), mu(mu)
{
    numLayers = (int)z.size();
    count = width*height;
    m_count = width*height*numLayers;
    blur = new Blur();

    allocate();
    multilayerPropagator(dx, lambda, n);
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
    
    int n[2] = {height, width};
    cufftPlanMany(&fftPlan, 2, n, NULL, 1, count, NULL, 1, count, CUFFT_C2C, 2);
}

void MultiLayer::multilayerPropagator(double dx, double lambda, double n){
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

    h_sum(m_count, &temp[m_count], sumArr);
    h_sum(count, &temp[count], &sumArr[N_BLOCKS]);
    
    scalef<<<1,1>>>(1,mu,sumArr,sumArr);
    simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&out[0]);
}

void MultiLayer::normalize(int cnt, double* arr){
    h_minimum(cnt, arr, sumArr);

    double temp;
    cudaMemcpy(&temp, sumArr, sizeof(double), cudaMemcpyDeviceToHost);
    offsetf<<<N_BLOCKS,N_THREADS>>>(cnt, -temp, arr, arr, true);

    h_maximum(cnt, arr, sumArr);
    contractf_p<<<N_BLOCKS,N_THREADS>>>(cnt, sumArr, arr, arr);
}

void MultiLayer::iterate(double *input, int iters, bool b_cost, bool warm){
    // Initialization of variables
    s = 1;
    if(b_cost){
        cudaMalloc(&cost, (1+iters)*sizeof(double));
        h_cost = (double*)malloc((iters+1)*sizeof(double));
    }

    //Copying the input image from host to device memory - computationally complex
    gpuErrchk(cudaMemcpy(image, input, count*sizeof(double), cudaMemcpyHostToDevice));
    blur->gaussianBlur(width,height, 5, 3, image, temporaryf, image);

    //Copying the device memory image to device memory guesses

    for(int i = 0; i < numLayers; i++){
        F2C<<<N_BLOCKS,N_THREADS>>>(count, image, &u[i*count]);
        if (!warm)
            F2C<<<N_BLOCKS,N_THREADS>>>(count, image, &guess[i*count]);
    }

    for(int iter = 0; iter < iters; iter++){
        //Calculating the current iteration model 
        propagate(Hq, u, temporary);

        //Calculation of Imodel and model arrays
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, temporary, model, Imodel);

        //Calculation of the optimal scaling parameter c
        h_sumOfProducts(count, image, Imodel, sumArr);
        h_sumOfProducts(count, Imodel, Imodel, &sumArr[N_BLOCKS]);
        contractf_p<<<1,1>>>(1, &sumArr[N_BLOCKS], sumArr, c);
        double t_cost[1];
        cudaMemcpy(t_cost, c, sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "Current optimal scaling factor is " << t_cost[0] << std::endl;

        //Cost calculation with sparsity constraint
        linear<<<N_BLOCKS,N_THREADS>>>(count, c, image, Imodel, temporaryf, false);

        if(b_cost){
            calculateCost(mu, temporaryf, guess, temporaryf, &cost[iter]);
            double t_cost[1];
            cudaMemcpy(t_cost, &cost[iter], sizeof(double), cudaMemcpyDeviceToHost);
            std::cout << t_cost[0] << std::endl;
        }

        //Calculating residues
        multiplyfc<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, model);
        extend<<<N_BLOCKS,N_THREADS>>>(count, numLayers, model, temporary);
        propagate(Hn, temporary, temporary);

        F2C<<<1,1>>>(1,c,newGuess);
        scale_p<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, u, temporary, newGuess, false);

        //Applying strict bounds
        for(int i = 0 ; i < numLayers ; i++){
            strictBounds<<<N_BLOCKS,N_THREADS>>>(count, &newGuess[count*i], rconstr[i*2], rconstr[i*2+1], iconstr[i*2], iconstr[i*2+1]);
        }

        //Applying soft thresholding bounds
        softBounds<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, mu, 0.5f);

        double s_new = 0.5*(1+std::sqrt(1+4*s*s));
        cufftDoubleComplex temp = make_cuDoubleComplex((s-1)/s_new,0);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, guess, temporary, false);
        scale<<<N_BLOCKS,N_THREADS>>>(m_count, temp, temporary, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(m_count, newGuess, temporary, u, true);

        s = s_new;
        cudaMemcpy(guess, newGuess, m_count*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
    
    }
    
    // Final cost calculation
    if(b_cost){
        propagate(Hq, u, newGuess);

        //Calculation of Imodel and model arrays
        modelFunc<<<N_BLOCKS,N_THREADS>>>(count, numLayers, 1.0f, 0, newGuess, model, Imodel);

        //Calculation of the optimal scaling parameter c
        h_sumOfProducts(count, image, Imodel, sumArr);
        h_sumOfProducts(count, Imodel, Imodel, &sumArr[N_BLOCKS]);
        contractf_p<<<1,1>>>(1, &sumArr[N_BLOCKS], sumArr, c);

        //Cost calculation with sparsity constraint
        linear<<<N_BLOCKS,N_THREADS>>>(count, c, image, Imodel, temporaryf, false);

        calculateCost(mu, temporaryf, guess, temporaryf, &cost[iters]);
        double t_cost[1];
        cudaMemcpy(t_cost, &cost[iters], sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << t_cost[0] << std::endl;

        gpuErrchk(cudaMemcpy(h_cost, cost, (iters+1)*sizeof(double), cudaMemcpyDeviceToHost));
        cudaFree(cost);
    }

    // Moving results to host memory
    // Adding one to get the light wavefront (otherwise we only have the disturbance by the particles and electrodes)
    offset<<<N_BLOCKS,N_THREADS>>>(m_count, 1.0f, 0.0f, guess, temporary);

    // Check if any error occured - important to note that untested kernels can lead to exceptions at cudaMemcpy calls
    gpuErrchk(cudaPeekAtLastError());
}

void MultiLayer::update(uint8_t* modulus, uint8_t* phase){
    // temporary contains the latest results in complex form
    
    // Processing the modulus of both layers
    absolute<<<N_BLOCKS,N_THREADS>>>(m_count,temporary,temporaryf);
    for(int i = 0; i < numLayers; i++){
        normalize(count, &temporaryf[i*count]);
    }
    D2u8<<<N_BLOCKS,N_THREADS>>>(m_count,temporaryf,modulus);

    // Processing the phase of both layers
    angle<<<N_BLOCKS,N_THREADS>>>(m_count,temporary,temporaryf);
    for(int i = 0; i < numLayers; i++){
        normalize(count, &temporaryf[i*count]);
    }
    D2u8<<<N_BLOCKS,N_THREADS>>>(m_count,temporaryf,phase);
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
}