#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cufft.h"
#include "lodepng.h"
#include "multilayer.h"
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

    vector<float> z{1e-3,2.75e-3};

    float dx = 1.55e-6;
    float n = 1.45;
    float lambda = 525e-9;

    float rconstr[4] = {-1,0,0,0};
    float iconstr[4] = {0,0,-1,1};

    const map<string, string> keywords{{"cmap","gray"}};

    vector<float> temp_image;
     
    temp_image = loadImage("Pics/hologram.png", width,height);
    static float image[2048*2048];
    copy(temp_image.begin(), temp_image.end(), image);

    MultiLayer *multilayer = new MultiLayer((int)width, (int)height, z, dx, lambda, n);
    multilayer->iterate(image, 6, 0.01, rconstr, iconstr);
    //for(int i = 0 ; i < width*height ; i ++)
    //    cout << multilayer->modulus[i] << "\n";

    float* m = multilayer->modulus;
    float* p = multilayer->phase;
    plt::imshow(m, height, width, 1, keywords);
    plt::title("Modulus for the 1st plane");
    plt::show();
    plt::title("Modulus for the 2nd plane");
    plt::imshow(&m[width*height], height, width, 1, keywords);
    plt::show();
    plt::title("Phase for 1st plane");
    plt::imshow(p, height, width, 1, keywords);
    plt::show();
    plt::title("Phase for 2nd plane");
    plt::imshow(&p[width*height], height, width, 1, keywords);
    plt::show();

    delete multilayer;



}