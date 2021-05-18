#ifndef APP_DATA_H
#define APP_DATA_H

#include "string.h"
#include "settings.h"
#include <vector>
#include <mutex>
#include <condition_variable>

using namespace std;
using json = nlohmann::json;

class AppData{
    private:
        void parseSettings(json j);
        json loadJson(const char* filename);
    public:
        AppData(const char* jsonFile = "params.json");
        ~AppData();

        string filename;
        string filetype;
        unsigned int width;
        unsigned int height;
        int iters0;
        int iters;
        double z;
        vector<double> rconstr;
        double dx;
        double n;
        double lambda;
        double mu;
        bool b_cost;
        vector<double> iconstr;
        double t;
        string result;

        mutex frameMtx;
        mutex displayMtx;

        condition_variable frameCv;
        condition_variable displayCv;

        double* inputFrame;
        uint8_t* d_phase;
        uint8_t* d_modulus;
        uint8_t* h_phase;
        uint8_t* h_modulus;
};

#endif