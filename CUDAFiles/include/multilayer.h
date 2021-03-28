#ifndef MULTILAYER_H
#define MULTILAYER_H

#include "cuda.h"
#include "cufft.h"
#include "blur.h"
#include <vector>
#include <iostream>

class MultiLayer
{
    private:
        cufftDoubleComplex *guess; // device memory planar guesses
        cufftDoubleComplex *newGuess; // device memory updated guesses for each plane
        cufftDoubleComplex *u; 
        cufftDoubleComplex *model; // device memory model for transformation between transmittance planes to 
        cufftDoubleComplex *temporary; // device memory placeholder for intermediate results
        double *temporaryf;

        int width;
        int height;
        int count;
        int m_count;
        Blur *blur;

        double *Imodel; // device memory norm of the model
        double s; // FISTA coefficient
        int numLayers;
        cufftHandle fftPlan;

        cufftComplex *Hq; // device memory backpropagation kernel
        cufftComplex *Hn; // device memory propagation kernel
        cufftComplex *propagation;
        double *image; // device memory real image
        double *cost;
        double* sumArr; // device memory for sum storing
        double* c;

        void multilayerPropagator(std::vector<double> z, double dx, double lambda, double n);
        void allocate();
        void calculateCost(double mu, double* model, cufftDoubleComplex* guess, double* temp, double* out);

    public:
        double *h_cost;

        double* modulus;
        double* phase;
        
        MultiLayer(int width, int height, std::vector<double> z, double dx, double lambda, double n);
        
        void iterate(double *input, int iters, double mu, double* rconstr, double* iconstr, bool b_cost);
        void propagate(cufftComplex* kernel, cufftDoubleComplex* input, cufftDoubleComplex* out);

        ~MultiLayer();

};

#endif