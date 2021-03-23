#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cufft.h"
#include "lodepng.h"
#include "multilayer.h"
#include "kernels.h"
#include "matplotlibcpp.h"
#include <chrono>

using namespace std;
namespace plt = matplotlibcpp;

vector<double> loadImage(const char* filename, unsigned int width, unsigned int height){
    vector<unsigned char> image;
    int size = (int)(height*width);
    vector<double> out(size, 0); 

    unsigned error = lodepng::decode(image, width, height, filename);
    if(error){
        cout << "Could not decode the image with error " << error << ": " << lodepng_error_text(error) << endl;
    }

    for(int i = 0; i < width*height; i++){
        out[i] = (double)image[i*4]/double(255);
    }

    return out;
}

void D2F(unsigned int count, double* input, float* output){
    for(int i = 0 ; i < count ; i++){
        output[i] = (float)input[i];
    }
}

int main(void)
{
    unsigned int width = 2048;
    unsigned int height = 2048;
    int iters = 6;

    vector<double> z{1e-3,2.75e-3};

    double dx = 1.55e-6;
    double n = 1.45;
    double lambda = 525e-9;

    double rconstr[4] = {-1,0,0,0};
    double iconstr[4] = {0,0,-1,1};

    const map<string, string> keywords{{"cmap","gray"}};

    vector<double> temp_image = loadImage("Pics/hologram.png", width,height);

    static double image[2048*2048];
    copy(temp_image.begin(), temp_image.end(), image);

    MultiLayer *multilayer = new MultiLayer((int)width, (int)height, z, dx, lambda, n);
    multilayer->iterate(image, iters, 0.01, rconstr, iconstr);
    for(int i = 0 ; i < iters+1 ; i++)
        cout << multilayer->h_cost[i] << "\n";
    
    double* m = multilayer->modulus;
    double* p = multilayer->phase;
    static float mf1[2048*2048];
    static float pf1[2048*2048];
    static float mf2[2048*2048];
    static float pf2[2048*2048];
    D2F(width*height, m, mf1);
    D2F(width*height, p, pf1);
    //D2F(width*height, &m[width*height], mf2);
    D2F(width*height, &p[width*height], pf2);
    
    plt::imshow(mf1, height, width, 1, keywords);
    plt::title("Modulus for the 1st plane");
    plt::show();
    /*
    plt::title("Modulus for the 2nd plane");
    plt::imshow(mf2, height, width, 1, keywords);
    plt::show();
    plt::title("Phase for 1st plane");
    plt::imshow(pf1, height, width, 1, keywords);
    plt::show();
    plt::title("Phase for 2nd plane");
    plt::imshow(pf2, height, width, 1, keywords);
    plt::show();
    */
    delete multilayer;



}