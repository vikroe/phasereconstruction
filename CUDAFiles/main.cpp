#include <stdio.h>
#include <iostream>
#include <fstream>
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
#include "json.hpp"

using namespace std;
using json = nlohmann::json;
namespace plt = matplotlibcpp;

json loadJson(const char* filename){
    ifstream in(filename);
    json j;
    in >> j;
    return(j);
}

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
    json j = loadJson("params.json");
    unsigned int width = j.at("width").get<unsigned int>();
    unsigned int height = j.at("height").get<unsigned int>();
    int iters = j.at("iters0").get<int>();
    int warmIters = j.at("iters").get<int>();
    vector<double> z = j.at("z").get<vector<double>>();
    vector<double> rconstr = j.at("rconstr").get<vector<double>>();
    vector<double> iconstr = j.at("iconstr").get<vector<double>>();
    double dx = j.at("dx").get<double>();
    double n = j.at("n").get<double>();
    double lambda = j.at("lambda").get<double>();
    double mu = j.at("mu").get<double>();
    bool b_cost = j.at("cost").get<bool>();


    vector<double> temp_image = loadImage("Pics/hologram.png", width,height);

    static double image[2048*2048];
    copy(temp_image.begin(), temp_image.end(), image);

    MultiLayer *multilayer = new MultiLayer((int)width, (int)height, z, rconstr, iconstr, mu, dx, lambda, n);
    auto start = chrono::steady_clock::now();
    multilayer->iterate(image, iters, b_cost);
    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration_cast<chrono::duration<double>>(end-start).count();
    printf("Time for the iterative FISTA algorithm: %f\n", elapsed);
    //for(int i = 0 ; i < iters+1 ; i++)
    //    cout << multilayer->h_cost[i] << "\n";
    
    double* m = multilayer->modulus;
    double* p = multilayer->phase;
    static float mf1[2048*2048];
    static float pf1[2048*2048];
    static float mf2[2048*2048];
    static float pf2[2048*2048];
    D2F(width*height, m, mf1);
    D2F(width*height, p, pf1);
    D2F(width*height, &m[width*height], mf2);
    D2F(width*height, &p[width*height], pf2);

    const map<string, string> keywords{{"cmap","gray"}};
    plt::subplot(2,2,1);
    plt::imshow(mf1, height, width, 1, keywords);
    plt::title("Plane 1 - modulus");
    plt::subplot(2,2,2);
    plt::title("Plane 2 - modulus");
    plt::imshow(mf2, height, width, 1, keywords);
    plt::subplot(2,2,3);
    plt::title("Plane 1 - phase");
    plt::imshow(pf1, height, width, 1, keywords);
    plt::subplot(2,2,4);
    plt::title("Plane 2 - phase");
    plt::imshow(pf2, height, width, 1, keywords);
    plt::show();
    delete multilayer;



}