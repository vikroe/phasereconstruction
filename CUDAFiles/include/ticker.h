#ifndef TICKER_H
#define TICKER_H

#include <chrono>
#include <string>

using namespace std;

class Ticker{
    private:
        bool set;
        chrono::time_point<chrono::steady_clock> start;
        chrono::time_point<chrono::steady_clock> end;
    public:
        Ticker();
        void tic();
        double toc(string msg = "");
        double tocc();
};

#endif