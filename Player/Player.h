
#pragma once 
#include <string>
#include <opencv2/opencv.hpp>
#include "ChannelObserver.h"

class Player : public ChannelObserver{

public:
    Player(std::string videoPath);
    Player(int deviceId);
    ~Player();

    virtual int snapshot(cv::Mat& frame, cv::Size unifiedSize);

private:
    std::string _videoPath;
    int _deviceId;

    cv::VideoCapture _cap;
};