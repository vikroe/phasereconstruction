#ifndef FISTA_H
#define FISTA_H

#include "cuda.h"
#include "cufft.h"
#include <vector>
#include <iostream>

class Fista
{
    private:
        double z;
        std::vector<double> rconstr;
        std::vector<double> iconstr;
        double mu;
        int width;
        int height;
        int count;
        bool b_cost;
        double m[1];
        double t;

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

        void normalize(int c, double* arr);
        void allocate();
        void calculateCost(double mu, double* model, cufftDoubleComplex* guess, double* temp, double* out);

    public:
        double *h_cost;
        
        Fista(
            double z,
            double mu,
            int width,
            int height,
            bool b_cost,
            double dx,
            double lambda,
            double n,
            double t
);
        
        void iterate(double *input, int iters, bool warm, bool scaling);
        void propagate(cufftComplex* kernel, cufftDoubleComplex* input, cufftDoubleComplex* out);
        //Used for parallelization
        void update(uint8_t* modulus, uint8_t* phase);

        ~Fista();

};

#endif