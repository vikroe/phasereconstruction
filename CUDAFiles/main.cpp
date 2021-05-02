#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include "ticker.h"
#include "multilayer.h"
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

void frameThread(AppData& appData){
    cout << appData.filename << endl;
    VideoParser video(appData.filename.c_str());
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
}

void retrievalThread(AppData& appData){
    Ticker *ticker = new Ticker();
    MultiLayer *multi = new MultiLayer((int)appData.width, 
                                    (int)appData.height, 
                                    appData.z, 
                                    appData.rconstr, 
                                    appData.iconstr, 
                                    appData.mu, 
                                    appData.dx, 
                                    appData.lambda, 
                                    appData.n);
    unsigned int count = appData.width*appData.height;
    double* image = new double[count];
    bool warm = false;
    while(!startingFrameCond && !windowOpened)
        ;
    ticker->tic();
    while(notFinished){
        unique_lock<mutex> flck(appData.frameMtx);
        memcpy(image, appData.inputFrame, count*sizeof(double));
        appData.frameCv.notify_one();
        flck.unlock();
        
        //ticker->tic();
        if(!warm)
            multi->iterate(image, appData.iters0, appData.b_cost, warm);
        else
            multi->iterate(image, appData.iters, appData.b_cost, warm);
        //ticker->toc("Current frame of multilayer FISTA took");
        
        unique_lock<mutex> dlck(appData.displayMtx);
        multi->update(appData.d_modulus, appData.d_phase);
        cudaMemcpy(appData.h_modulus, appData.d_modulus, sizeof(uint8_t)*count*appData.z.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(appData.h_phase, appData.d_phase, sizeof(uint8_t)*count*appData.z.size(), cudaMemcpyDeviceToHost);
        appData.displayCv.notify_one();
        dlck.unlock();
        if(!startingDisplayCond)
            startingDisplayCond = true;
        if(!warm)
            warm = true;
        ticker->toc("[RETRIEVAL] This frame took the retrievalThread ");
    }
    delete multi;
}

void displayThread(AppData& appData){
    unsigned int count = appData.width*appData.height;
    const map<string, string> keywords{{"cmap","gray"}};
    Ticker* dticker = new Ticker();

    while(!startingDisplayCond)
        ;
    cv::namedWindow("Visualization", cv::WINDOW_NORMAL);
    cv::resizeWindow("Visualization", appData.width, appData.height);
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
        if (ret_key == 27 || ret_key == 'x') notFinished = false;
    }
    cv::destroyWindow("Visualization");
}

int main(void)
{
    static AppData appData;
    std::cout << appData.filename << "what" << endl;

    const unsigned int count = appData.width*appData.height;
    notFinished = true;
    windowOpened = false;

    thread frameThr (frameThread, std::ref(appData));
    thread retrievalThr (retrievalThread, std::ref(appData));
    thread displayThr (displayThread, std::ref(appData));

    frameThr.join();
    retrievalThr.join();
    displayThr.join();
    //for(int i = 0 ; i < iters+1 ; i++)
    //    cout << multilayer->h_cost[i] << "\n";
    /*
    double* m = multilayer->modulus;
    double* p = multilayer->phase;
    D2F(width*height, m, mf1);
    D2F(width*height, p, pf1);
    D2F(width*height, &m[width*height], mf2);
    D2F(width*height, &p[width*height], pf2);
    */
    


}