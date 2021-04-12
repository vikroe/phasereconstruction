#ifndef VIDEO_PARSER_H
#define VIDEO_PARSER_H

#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

class VideoParser{
    private:
        VideoCapture capture;
        Size size;
    public:
        VideoParser();
        VideoParser(const char* filename);
        void openFile(const char* filename);
        int loadFrame(const unsigned int width, const unsigned int height, double* oframe);
        ~VideoParser();
        unsigned int getHeight();
        unsigned int getWidth();
        unsigned int getFrameCount();
        unsigned int getCurrentFrame();
};

#endif