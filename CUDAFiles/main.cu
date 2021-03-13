#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cufft.h"
#include "lodepng.h"
#include "kernels.h"
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

vector<float> loadImage(const char* filename, unsigned int width, unsigned int height){
    vector<unsigned char> image;
    int size = (int)(height*width);
    vector<float> out(size, 0); 

    unsigned error = lodepng::decode(image, width, height, filename);
    if(error){
        cout << "Could not decode the image with error " << error << ": " << lodepng_error_text(error) << endl;
    }

    for(int i = 0; i < width*height; i++){
        out[i] = (float)image[i*4]/float(255);
    }

    return out;
} 

int main(void)
{
    unsigned int width = 2048;
    unsigned int height = 2048;

    float z_electrodes = 2.5e-3;
    float z_particles = 2.7e-3;

    float dx = 1.55e-6;
    float n = 1.45;
    float lambda = 525e-9;

    vector<float> temp_image;
     
    temp_image = loadImage("Pics/electrodes.png", width,height);
    static float image[2048*2048];
    copy(temp_image.begin(), temp_image.end(), image);

    printf("%f\n", image[0]);

    cufftComplex* Hq_electrodes;
    cufftComplex* Hq_particles;
    cudaMalloc(&Hq_electrodes, width*height*sizeof(cufftComplex));
    cudaMalloc(&Hq_particles, width*height*sizeof(cufftComplex));
    propagator<<<N_BLOCKS,N_THREADS>>>(height, width, z_electrodes, dx, n, lambda, Hq_electrodes);
    propagator<<<N_BLOCKS,N_THREADS>>>(height, width, z_particles, dx, n, lambda, Hq_particles);
    printf("Managed this!\n");
    cudaFree(Hq_electrodes);
    cudaFree(Hq_particles);

    plt::imshow(image, height, width, 1);
    plt::show();


}