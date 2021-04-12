#include <iostream>
#include <vector>
#include "videoParser.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

VideoParser::VideoParser(){
    std::cout << "Could compile" << "\n";
}

VideoParser::VideoParser(const char *filename){
    capture.open(filename);
    if (!capture.isOpened())
        std::cout << "Could not open video file " << filename << "\n";
}

void VideoParser::openFile(const char *filename){
    capture.open(filename);
    if (!capture.isOpened())
        std::cout << "Could not open video file " << filename << "\n";
}

int VideoParser::loadFrame(const unsigned int width, const unsigned int height, double* oframe){
    Mat frame;
    capture.read(frame);
    if(frame.empty()){
        std::cout << "Loaded an empty frame!" << "\n";
        return 101;
    }

    for(int i = 0; i < width; i++){
        for(int j = 0; j < height; j++){
            oframe[i*width + j] = (double)(frame.at<Vec3b>(i,j)[0])/255.0f;
        
        }
    }
    return 0;
    
}

unsigned int VideoParser::getHeight(){
    return (unsigned int)capture.get(CAP_PROP_FRAME_HEIGHT);
}

unsigned int VideoParser::getWidth(){
    return (unsigned int)capture.get(CAP_PROP_FRAME_WIDTH);
}

//Does not always work?
unsigned int VideoParser::getFrameCount(){
    return (unsigned int)capture.get(CAP_PROP_FRAME_COUNT);
}

unsigned int VideoParser::getCurrentFrame(){
    return (unsigned int)capture.get(CAP_PROP_POS_FRAMES);
}


VideoParser::~VideoParser(){
    if(capture.isOpened())
        capture.release();
}

