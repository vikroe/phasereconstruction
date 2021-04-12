#include "utils.h"
#include <vector>
#include <iostream>
#include "lodepng.h"

void D2F(unsigned int count, double* input, float* output){
    for(int i = 0 ; i < count ; i++){
        output[i] = (float)input[i];
    }
}

void U82D(unsigned int count, unsigned char* in, double* out){
    for(int i = 0 ; i < count ; i++){
        out[i] = (double)(in[i])/255.0f;
    }
}

std::vector<double> loadImage(const char* filename, unsigned int width, unsigned int height){
    std::vector<unsigned char> image;
    int size = (int)(height*width);
    std::vector<double> out(size, 0); 

    unsigned error = lodepng::decode(image, width, height, filename);
    if(error){
        std::cout << "Could not decode the image with error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    for(int i = 0; i < width*height; i++){
        out[i] = (double)image[i*4]/double(255);
    }

    return out;
}
