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
        float *temporaryf;

        int width;
        int height;
        int count;
        int m_count;

        float *Imodel; // device memory norm of the model
        float s; // FISTA coefficient
        int numLayers;
        cufftHandle fftPlan;

        cufftComplex *Hq; // device memory backpropagation kernel
        cufftComplex *Hn; // device memory propagation kernel
        cufftComplex *image; // device memory complex image
        float *imagef; // device memory real image
        float *cost;
        float* sumArr; // device memory for sum storing
        float* c;

        void multilayerPropagator(std::vector<float> z, float dx, float lambda, float n);

    public:
        float *h_cost;

        float* modulus;
        float* phase;
        
        MultiLayer(int width, int height, std::vector<float> z, float dx, float lambda, float n);
        
        void iterate(float *input, int iters, float mu, float* rconstr, float* iconstr);
        void propagate(cufftComplex* kernel, cufftComplex* input, cufftComplex* out);

        ~MultiLayer();

};

#endif