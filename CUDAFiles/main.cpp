#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include "ticker.h"
#include "fista.h"
#include "kernels.h"
#include "videoParser.h"
#include "utils.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cuda.h"
#include <cuda_runtime.h>
#include "cudaDebug.h"
#include "appData.h"

using namespace std;

bool startingFrameCond;
bool startingDisplayCond;
bool notFinished;
bool windowOpened;
bool quit;

void frameThread(AppData& appData){
    cout << appData.filename << endl;
    VideoParser video(appData);
    while(!windowOpened)
        ;
    while (video.getCurrentFrame() < 100){
        unique_lock<mutex> lck(appData.frameMtx);
        if (video.loadFrame(appData.width, appData.height, appData.inputFrame) != 0){
            appData.frameCv.wait(lck);
            lck.unlock();
            cout << "[FRAME] End of video reached! Closing frame thread.\n";
            break;
        }
        if((int)video.getCurrentFrame() == 1)
            startingFrameCond = true;
        appData.frameCv.wait(lck);
        cout << "[FRAME] Gathering " << video.getCurrentFrame() << ". frame." << endl;
        lck.unlock();
    }
    notFinished = false;
    while(!quit)
        ;
}

void retrievalThread(AppData& appData){
    Ticker *ticker = new Ticker();
    Fista *fista = new Fista(appData.z,
                            appData.rconstr,
                            appData.iconstr,
                            appData.mu,
                            appData.width,
                            appData.height,
                            appData.b_cost,
                            appData.dx,
                            appData.lambda,
                            appData.n);
    unsigned int count = appData.width*appData.height;
    double* image = new double[count];
    bool warm = false;
    while(!startingFrameCond || !windowOpened)
        ;
    ticker->tic();
    while(notFinished){
        unique_lock<mutex> flck(appData.frameMtx);
        memcpy(image, appData.inputFrame, count*sizeof(double));
        appData.frameCv.notify_one();
        flck.unlock();
        
        //ticker->tic();
        if(!warm)
            fista->iterate(image, appData.iters0, warm);
        else
            fista->iterate(image, appData.iters, warm);
        //ticker->toc("Current frame of fistalayer FISTA took");
        
        unique_lock<mutex> dlck(appData.displayMtx);
        fista->update(appData.d_modulus, appData.d_phase);
        cudaMemcpy(appData.h_modulus, appData.d_modulus, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        cudaMemcpy(appData.h_phase, appData.d_phase, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        appData.displayCv.notify_one();
        dlck.unlock();
        if(!warm)
            warm = true;
        ticker->toc("[RETRIEVAL] This frame took the retrievalThread ");
    }
    delete fista;
    while(!quit)
        ;
}

void displayThread(AppData& appData){
    unsigned int count = appData.width*appData.height;
    const map<string, string> keywords{{"cmap","gray"}};
    Ticker* dticker = new Ticker();
    cv::namedWindow("Visualization", cv::WINDOW_NORMAL);
    cv::resizeWindow("Visualization", appData.width, appData.height);
    cv::waitKey(1);
    windowOpened = true;
    while(notFinished){
        dticker->tic();
        unique_lock<mutex> dlck(appData.displayMtx);
        appData.displayCv.wait(dlck);
        dlck.unlock();
        const cv::Mat p1(cv::Size(appData.width, appData.height), CV_8U, appData.h_phase);
        const cv::Mat m1(cv::Size(appData.width, appData.height), CV_8U, appData.h_modulus);
        dticker->toc("[DISPLAY] This upload of frame to cv::Mat took");
        cv::imshow("Visualization", m1);
        char ret_key = (char) cv::waitKey(1);
        if (ret_key == 27 || ret_key == 'x') {
            notFinished = false;
            quit = true;
        }
    }
    quit = true;
    cv::destroyWindow("Visualization");
}

int main(void)
{
    static AppData appData;

    const unsigned int count = appData.width*appData.height;

    if(appData.filetype == "AVI"){
        notFinished = true;
        windowOpened = false;
        startingDisplayCond = false;
        startingFrameCond = false;
        quit = false;

        thread frameThr (frameThread, std::ref(appData));
        thread retrievalThr (retrievalThread, std::ref(appData));
        thread displayThr (displayThread, std::ref(appData));

        frameThr.join();
        retrievalThr.join();
        displayThr.join();
    }
    else if(appData.filetype == "PNG"){
        vector<double> v_image = loadImage(appData.filename.c_str(), appData.width, appData.height);
        double* image = (double*)malloc(sizeof(double)*appData.height*appData.width);
        copy(v_image.begin(), v_image.end(), image);
        Fista *fista = new Fista(appData.z,
                            appData.rconstr,
                            appData.iconstr,
                            appData.mu,
                            appData.width,
                            appData.height,
                            appData.b_cost,
                            appData.dx,
                            appData.lambda,
                            appData.n);
        fista->iterate(image, appData.iters0, false);
        fista->update(appData.d_modulus, appData.d_phase);
        cudaMemcpy(appData.h_modulus, appData.d_modulus, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        cudaMemcpy(appData.h_phase, appData.d_phase, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        cv::namedWindow("Visualization", cv::WINDOW_NORMAL);
        cv::resizeWindow("Visualization", appData.width, appData.height);
        const cv::Mat p1(cv::Size(appData.width, appData.height), CV_8U, appData.h_phase);
        const cv::Mat m1(cv::Size(appData.width, appData.height), CV_8U, appData.h_modulus);
        cv::imshow("Visualization", m1);
        cv::waitKey(0);
        cv::destroyWindow("Visualization");
        free(image);
    }

    appData.~AppData();
}