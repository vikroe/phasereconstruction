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
        std::vector<double> z;
        std::vector<double> rconstr;
        std::vector<double> iconstr;
        double mu;
        int numLayers;
        int width;
        int height;
        int count;
        int m_count;

        Blur *blur;

        cufftDoubleComplex *guess; // device memory planar guesses
        cufftDoubleComplex *newGuess; // device memory updated guesses for each plane
        cufftDoubleComplex *u; 
        cufftDoubleComplex *model; // device memory model for transformation between transmittance planes to 
        cufftDoubleComplex *temporary; // device memory placeholder for intermediate results
        cufftComplex *Hq; // device memory backpropagation kernel
        cufftComplex *Hn; // device memory propagation kernel
        cufftComplex *propagation;
        double *temporaryf;
        double *Imodel; // device memory norm of the model
        double *image; // device memory real image
        double *cost;
        double* sumArr; // device memory for sum storing
        double* c;
        double s; // FISTA coefficient

        cufftHandle fftPlan;

        void multilayerPropagator(double dx, double lambda, double n);
        void allocate();
        void calculateCost(double mu, double* model, cufftDoubleComplex* guess, double* temp, double* out);

    public:
        double *h_cost;

        double* modulus;
        double* phase;
        
        MultiLayer(
            int width,
            int height, 
            std::vector<double> z, 
            std::vector<double> rconstr, 
            std::vector<double> iconstr, 
            double mu, 
            double dx, 
            double lambda, 
            double n);
        
        void iterate(double *input, int iters, bool b_cost);
        void propagate(cufftComplex* kernel, cufftDoubleComplex* input, cufftDoubleComplex* out);

        ~MultiLayer();

};

#endif