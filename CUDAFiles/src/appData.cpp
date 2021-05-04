#include "appData.h"
#include "json.hpp"
#include <fstream>
#include "cuda.h"
#include <cuda_runtime.h>

AppData::AppData(const char* jsonFile){
    json j = loadJson(jsonFile);
    parseSettings(j);
    cudaMalloc(&d_modulus, width*height*sizeof(uint8_t));
    cudaMalloc(&d_phase, width*height*sizeof(uint8_t));
    cudaMallocHost(&inputFrame, width*height*sizeof(double));
    cudaMallocHost(&h_modulus, width*height*sizeof(uint8_t));
    cudaMallocHost(&h_phase, width*height*sizeof(uint8_t));
}

json AppData::loadJson(const char* filename){
    std::ifstream in(filename);
    json j;
    in >> j;
    return(j);
}

void AppData::parseSettings(json j){
    filename = j.at("filename").get<std::string>();
    filetype = j.at("filetype").get<std::string>();
    width =    j.at("width").get<unsigned int>();
    height =   j.at("height").get<unsigned int>();
    iters0 =   j.at("iters0").get<int>();
    iters =    j.at("iters").get<int>();
    z =        j.at("z").get<double>();
    rconstr =  j.at("rconstr").get<std::vector<double>>();
    dx =       j.at("dx").get<double>();
    n =        j.at("n").get<double>();
    lambda =   j.at("lambda").get<double>();
    mu =       j.at("mu").get<double>();
    b_cost =   j.at("cost").get<bool>();
    iconstr =  j.at("iconstr").get<std::vector<double>>();
}

AppData::~AppData(){
    cudaFree(d_modulus);
    cudaFree(d_phase);
    cudaFreeHost(inputFrame);
    cudaFreeHost(h_modulus);
    cudaFreeHost(h_phase);
}