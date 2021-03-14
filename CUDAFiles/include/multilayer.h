#ifndef MULTILAYER
#define MULTILAYER

#include "cuda.h"
#include "cufft.h"
#include <vector>
#include <iostream>

class MultiLayer
{
    private:
        cufftComplex *res; // device memory residues
        cufftComplex *guess; // device memory planar guesses
        cufftComplex *newGuess; // device memory updated guesses for each plane
        cufftComplex *u; 
        cufftComplex *model; // device memory model for transformation between transmittance planes to 
        cufftComplex *temporary; // device memory placeholder for intermediate results

        int width;
        int height;

        float *Imodel; // device memory norm of the model
        float s; // FISTA coefficient
        int numLayers;
        int numBlocks;
        cufftHandle fftPlan;

        cufftComplex *Hq; // device memory backpropagation kernel
        cufftComplex *Hn; // device memory propagation kernel
        cufftComplex *image; // device memory complex image
        float *imagef; // device memory real image
        float *cost;
        float* ImodelSum; // device memory sum of Imodel
        float* imageSum; // device memory sum of image

    public:
        float *h_cost;
        MultiLayer(int width, int height, std::vector<float> z, float dx, float lambda, float n);
        void iterate(float *input, int iters);
        void propagate(cufftComplex* kernel, cufftComplex* input, cufftComplex* out);

        void multilayerPropagator(std::vector<float> z, float dx, float lambda, float n);

        ~MultiLayer();

};

#endif