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


double Ticker::toc(string msg){
    if(!set){
        printf("tic() has to be called first!\n");
    }
    else{
        end = chrono::steady_clock::now();
        double elapsed_seconds = chrono::duration_cast<
            chrono::duration<double> >(end - start).count();
        printf("%s %f s\n", msg.c_str(), elapsed_seconds);
        start = end;
        return elapsed_seconds;
    }
    return 0;
}


double Ticker::tocc(){
    if(!set){
        printf("tic() has to be called first!\n");
    }
    else{
        end = chrono::steady_clock::now();
        double elapsed_seconds = chrono::duration_cast<
            chrono::duration<double> >(end - start).count();
        start = end;
        return elapsed_seconds;
    }
    return 0;
}