#include "ticker.h"
#include <iostream>
#include <chrono>
#include <string.h>

using namespace std;

Ticker::Ticker(){
    set = false;
}

void Ticker::tic(){
    start = chrono::steady_clock::now();
    set = true;
}

void Ticker::toc(string msg){
    if(!set){
        cout << "tic() has to be called first!" << endl;
    }
    else{
        end = chrono::steady_clock::now();
        double elapsed_seconds = chrono::duration_cast<
            chrono::duration<double> >(end - start).count();
        cout << msg << " " << elapsed_seconds << " s" << endl;
        start = end;
    }
}