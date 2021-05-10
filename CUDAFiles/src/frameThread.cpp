#include "frameThread.h"
#include "appData.h"
#include <iostream>
#include "videoParser.h"
#include <mutex>
#include <condition_variable>

using namespace std;

void frameThread(AppData& appData){
    cout << appData.filename << endl;
    VideoParser video(appData.filename.c_str());
    while (video.getCurrentFrame() < 100){
        unique_lock<mutex> lck(appData.frameMtx);
        if (video.loadFrame(appData.width, appData.height, appData.inputFrame) != 0){
            appData.frameCv.wait(lck);
            lck.unlock();
            cout << "End of video reached! Closing frame thread.\n";
            break;
        }
        cout << "Gathering " << video.getCurrentFrame() << ". frame." << endl;
        if((int)video.getCurrentFrame() == 1)
            startingFrameCond = true;
        appData.frameCv.wait(lck);
        lck.unlock();
    }
    notFinished = false;
}