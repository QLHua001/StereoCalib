#pragma once
#include <opencv2/opencv.hpp>

class ChannelObserver{

public:
    virtual int snapshot(cv::Mat& frame) = 0;
};