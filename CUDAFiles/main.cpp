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
#include <ctime>
#include <iomanip>

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
    while (video.getCurrentFrame() < 451){
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
                            appData.mu,
                            appData.width,
                            appData.height,
                            appData.b_cost,
                            appData.dx,
                            appData.lambda,
                            appData.n,
                            appData.t);
    unsigned int count = appData.width*appData.height;
    double* image = new double[count];
    bool warm = false;
    bool scaling = false;
    int iters = appData.iters0;
    while(!startingFrameCond || !windowOpened)
        ;
    int iteration_count = 0;
    double average_cycle = 0;
    double variance = 0;
    double current_time = 0;
    while(notFinished){
        unique_lock<mutex> flck(appData.frameMtx);
        memcpy(image, appData.inputFrame, count*sizeof(double));
        appData.frameCv.notify_one();
        flck.unlock();
        ticker->tic();
        
        fista->iterate(image, iters, warm, scaling);

        fista->update(appData.d_modulus, appData.d_phase);
        cudaMemcpy(appData.h_modulus, appData.d_modulus, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        cudaMemcpy(appData.h_phase, appData.d_phase, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);

        appData.displayCv.notify_one();
        
        if(!warm){
            if(appData.warm)
                warm = true;
            iters = appData.iters;
        }
        current_time = ticker->toc("[RETRIEVAL] This frame took the retrievalThread ");
        if(iteration_count > 0){
            double new_average_cycle = (average_cycle*(iteration_count-1)+current_time)/(double)(iteration_count);
            variance = ((iteration_count-1)*variance)/(double)(iteration_count) + (current_time - average_cycle)*(current_time - new_average_cycle)/(double)(iteration_count);
            average_cycle = new_average_cycle;
        }
        iteration_count += 1;
    }
    delete fista;
    while(!quit)
        ;
    cout << "Average retrieval thread cycle comes up to " << average_cycle << " s." << endl;
    cout << "The variance of the average cycle comes up to " << variance << " s." << endl;
}

void displayThread(AppData& appData){
    unsigned int count = appData.width*appData.height;
    const map<string, string> keywords{{"cmap","gray"}};
    Ticker* dticker = new Ticker();
    cv::namedWindow("Visualization", cv::WINDOW_NORMAL);
    cv::resizeWindow("Visualization", appData.width, appData.height);
    cv::waitKey(1);
    cv::VideoWriter video_writer;
    if (appData.record) {
        // Define the codec and create VideoWriter object.The output is stored in '%H%M%S_%d%m%Y.avi.avi' file. 
        time_t t = time(nullptr);
        tm tm = *localtime(&t);
        stringstream filename;
        filename << put_time(&tm, "./Results/%H%M%S_%d%m%Y.avi");
        video_writer.open(filename.str(), cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(appData.width, appData.height), false);

        if (!video_writer.isOpened()) {
            fprintf(stdout, "ERROR: failed to open the video file.\n");
            appData.record = false;
        }
    }
    windowOpened = true;
    int iteration_count = 0;
    while(notFinished){
        dticker->tic();
        unique_lock<mutex> dlck(appData.displayMtx);
        appData.displayCv.wait(dlck);
        dlck.unlock();
        const cv::Mat p1(cv::Size(appData.width, appData.height), CV_8U, appData.h_phase);
        const cv::Mat m1(cv::Size(appData.width, appData.height), CV_8U, appData.h_modulus);
        dticker->toc("[DISPLAY] This upload of frame to cv::Mat took");
        if (appData.record) {
            video_writer.write(m1);
        }
        cv::imshow("Visualization", m1);
        char ret_key = (char) cv::waitKey(1);
        if (ret_key == 27 || ret_key == 'x') {
            notFinished = false;
            quit = true;
        }
        iteration_count += 1;
        cout << "DISPLAY iteration " << iteration_count << endl;
    }
    quit = true;
    notFinished = false;
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
                            appData.mu,
                            appData.width,
                            appData.height,
                            appData.b_cost,
                            appData.dx,
                            appData.lambda,
                            appData.n,
                            appData.t);
        fista->iterate(image, appData.iters0, false, false);
        fista->update(appData.d_modulus, appData.d_phase);
        cudaMemcpy(appData.h_modulus, appData.d_modulus, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        cudaMemcpy(appData.h_phase, appData.d_phase, sizeof(uint8_t)*count, cudaMemcpyDeviceToHost);
        cv::namedWindow("Visualization", cv::WINDOW_NORMAL);
        cv::resizeWindow("Visualization", appData.width, appData.height);
        const cv::Mat p1(cv::Size(appData.width, appData.height), CV_8U, appData.h_phase);
        const cv::Mat m1(cv::Size(appData.width, appData.height), CV_8U, appData.h_modulus);
        cv::imwrite(appData.result, m1);
        cv::imshow("Visualization", m1);
        cv::waitKey(0);
        cv::destroyWindow("Visualization");
        free(image);
    }

    appData.~AppData();
}