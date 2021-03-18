#include "multilayer.h"
#include "cuda.h"
#include "cufft.h"
#include "kernels.h"
#include <vector>
#include <iostream>
#include <math.h>

MultiLayer::MultiLayer(int width, int height, std::vector<float> z, float dx, float lambda, float n) :width{width}, height{height}
{
    numLayers = (int)z.size();
    numBlocks = (width*height/2 + N_BLOCKS -1)/N_BLOCKS;

    cudaMalloc(&Hq, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&Hn, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&res, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&guess, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&newGuess, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&u, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&temporary, numLayers*width*height*sizeof(cufftComplex));
    cudaMalloc(&temporaryf, 2*numLayers*width*height*sizeof(float));

    cudaMalloc(&image, width*height*sizeof(cufftComplex));
    cudaMalloc(&imagef, width*height*sizeof(float));
    cudaMalloc(&sumArr, 2*N_BLOCKS*sizeof(float));
    cudaMalloc(&c, sizeof(float));
    
    cudaMalloc(&model, width*height*sizeof(cufftComplex));
    cudaMalloc(&Imodel, width*height*sizeof(float));

    cufftPlan2d(&fftPlan, width, height, CUFFT_C2C);

    multilayerPropagator(z, dx, lambda, n);

}

void MultiLayer::multilayerPropagator(std::vector<float> z, float dx, float lambda, float n){
    cufftComplex *placeHolder;
    for(int i = 0; i < numLayers; i++){
        placeHolder = &Hq[i*width*height];
        propagator<<<N_BLOCKS,N_THREADS>>>(width, height, z[i], dx, n, lambda, placeHolder);
    }
}

void MultiLayer::propagate(cufftComplex* kernel, cufftComplex* input, cufftComplex* out){
    cufftExecC2C(fftPlan, input, out, CUFFT_FORWARD);
    multiply<<<N_BLOCKS, N_THREADS>>>(width,height,kernel,out);
    cufftExecC2C(fftPlan, out, out, CUFFT_INVERSE);
}

void MultiLayer::iterate(float *input, int iters, float mu, float* rconstr, float* iconstr, float* modulus, float* phase){
    // Initialization of variables
    s = 1;
    float t = 0.5;
    int count = width*height;
    float *fplaceHolder;
    cufftComplex *placeHolder;
    cufftComplex *HplaceHolder;

    conjugate<<<N_BLOCKS,N_THREADS>>>(width*numLayers, height, Hq, Hn);

    //Allocating the device memory array for cost at each iteration
    cudaMalloc(&cost, (1+iters)*sizeof(float));

    //Copying the input image from host to device memory - computationally complex
    cudaMemcpy(imagef, input, width*height*sizeof(float), cudaMemcpyHostToDevice);
    F2C<<<N_BLOCKS, N_THREADS>>>(width, height, imagef, image);

    //Copying the device memory image to device memory guesses
    for(int i = 0; i < numLayers; i++){
        cudaMemcpy(&guess[i*width*height], image, width*height*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
        cudaMemcpy(&u[i*width*height], image, width*height*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    }

    for(int iter = 0; iter < iters; iter++){

        //Calculating the current iteration model
        for(int i = 0; i < numLayers; i++){
            placeHolder = &temporary[i*width*height];
            HplaceHolder = &Hq[i*width*height];
            propagate(HplaceHolder, image, placeHolder);
        }
        modelFunc<<<N_BLOCKS,N_THREADS>>>(width, height, numLayers, 1.0f, 0, temporary, model);
        ImodelFunc<<<N_BLOCKS,N_THREADS>>>(width,height,model,Imodel);

        fplaceHolder = temporaryf;
        multiplyf<<<N_BLOCKS, N_THREADS>>>(width, height, Imodel, imagef, fplaceHolder);
        sum<<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, sumArr);
        sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS, sumArr, sumArr);
        fplaceHolder = &temporaryf[count];
        multiplyf<<<N_BLOCKS, N_THREADS>>>(width, height, Imodel, Imodel, fplaceHolder);
        sum<<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, &sumArr[N_BLOCKS]);
        sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS, &sumArr[N_BLOCKS], &sumArr[N_BLOCKS]);

        simpleDivision<<<1,1>>>(sumArr, &sumArr[N_BLOCKS], c);

        linear<<<N_BLOCKS,N_THREADS>>>(width,height,c,imagef,Imodel, temporaryf, false);

        absolute<<<N_BLOCKS,N_THREADS>>>(numLayers*width,height,guess,&temporaryf[2*count]);
        sum<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(float)>>>(numLayers*count,&temporaryf[2*count],sumArr);
        sum<<<1,N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS,sumArr,sumArr);

        square<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, &temporaryf[count]);
        sum<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(float)>>>(numLayers*count,&temporaryf[count],&sumArr[N_BLOCKS]);
        sum<<<1,N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS,&sumArr[N_BLOCKS],&sumArr[N_BLOCKS]);

        //Cost calculation with sparsity constraint
        cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
        simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&cost[iter]);
        
        multiplyfc<<<N_BLOCKS,N_THREADS>>>(count*numLayers, temporaryf, temporary);
        for(int i = 0; i < numLayers; i++){
            placeHolder = &res[i*count];
            HplaceHolder = &Hn[i*count];
            propagate(HplaceHolder, temporary, placeHolder);
        }

        cMultiplyf<<<1,1>>>(1,(2*t),c,c);
        cMultiplyfcp<<<N_BLOCKS,N_THREADS>>>(numLayers*count, c, res, temporary);
        add<<<N_BLOCKS,N_THREADS>>>(count*numLayers, u, temporary, newGuess, false);

        //Applying strict bounds
        for(int i = 0 ; i < numLayers ; i++){
            placeHolder = &newGuess[count*i];
            strictBounds<<<N_BLOCKS,N_THREADS>>>(count, placeHolder, rconstr[i*2], rconstr[i*2+1], iconstr[i*2], iconstr[i*2+1]);
        }

        //Applying soft thresholding bounds
        softBounds<<<N_BLOCKS,N_THREADS>>>(count*numLayers,newGuess,mu,t);

        float s_new = 0.5*(1+std::sqrt(1+4*s*s));
        float temp = (s-1)/s_new;
        add<<<N_BLOCKS,N_THREADS>>>(count*numLayers,newGuess,guess, temporary,false);
        cMultiplyfc<<<N_BLOCKS,N_THREADS>>>(count,temp,temporary,temporary);
        add<<<N_BLOCKS,N_THREADS>>>(count*numLayers, newGuess, temporary, u, false);

        s = s_new;
        cudaMemcpy(guess, newGuess, numLayers*width*height*sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    
    }

    // Final cost calculation
    //Calculating the current iteration model
    for(int i = 0; i < numLayers; i++){
        placeHolder = &temporary[i*width*height];
        HplaceHolder = &Hq[i*width*height];
        propagate(HplaceHolder, image, placeHolder);
    }
    modelFunc<<<N_BLOCKS,N_THREADS>>>(width, height, numLayers, 1.0f, 0, temporary, model);
    ImodelFunc<<<N_BLOCKS,N_THREADS>>>(width,height,model,Imodel);

    fplaceHolder = temporaryf;
    multiplyf<<<N_BLOCKS, N_THREADS>>>(width, height, Imodel, imagef, fplaceHolder);
    sum<<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, sumArr);
    sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS, sumArr, sumArr);
    fplaceHolder = &temporaryf[count];
    multiplyf<<<N_BLOCKS, N_THREADS>>>(width, height, Imodel, Imodel, fplaceHolder);
    sum<<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(float)>>>(count, fplaceHolder, &sumArr[N_BLOCKS]);
    sum<<<1, N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS, &sumArr[N_BLOCKS], &sumArr[N_BLOCKS]);

    simpleDivision<<<1,1>>>(sumArr, &sumArr[N_BLOCKS], c);

    linear<<<N_BLOCKS,N_THREADS>>>(width,height,c,imagef,Imodel, temporaryf, false);

    absolute<<<N_BLOCKS,N_THREADS>>>(numLayers*width,height,guess,&temporaryf[2*count]);
    sum<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(float)>>>(numLayers*count,&temporaryf[2*count],sumArr);
    sum<<<1,N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS,sumArr,sumArr);

    square<<<N_BLOCKS,N_THREADS>>>(count, temporaryf, &temporaryf[count]);
    sum<<<N_BLOCKS,N_THREADS, N_THREADS*sizeof(float)>>>(numLayers*count,&temporaryf[count],&sumArr[N_BLOCKS]);
    sum<<<1,N_BLOCKS, N_BLOCKS*sizeof(float)>>>(N_BLOCKS,&sumArr[N_BLOCKS],&sumArr[N_BLOCKS]);

    //Cost calculation with sparsity constraint
    cMultiplyf<<<1,1>>>(1,mu,sumArr,sumArr);
    simpleSum<<<1,1>>>(&sumArr[N_BLOCKS],sumArr,&cost[iters]);

    // Moving results to host memory

    absolute<<<N_BLOCKS,N_THREADS>>>(width*numLayers,height,guess,temporaryf);
    cudaMemcpy(modulus,temporaryf,width*numLayers*height,cudaMemcpyDeviceToHost);
    angle<<<N_BLOCKS,N_THREADS>>>(count*numLayers,guess,temporaryf);
    cudaMemcpy(phase,temporaryf,width*numLayers*height,cudaMemcpyDeviceToHost);

    // Deallocation of variables
    cudaFree(cost);
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
}