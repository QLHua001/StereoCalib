#pragma once

#include "QCamCalib/QCamCalib.h"
#include "StereoCalib/StereoCalib.h"
#include "Player/Controller.h"
#include "Collector/Collector.h"

class CalibSys{

public:
    enum InputType{
        TYPE_VIDEO,
        TYPE_CAMERA
    };

public:
    CalibSys(InputType inputType, std::vector<int>& cameras, cv::Size imageSize);
    CalibSys(InputType InputType, std::vector<std::string>& videos, cv::Size imageSize);
    ~CalibSys();

    void init();

    void run();

private:
    InputType _inputType;
    std::vector<std::string> _videos;
    std::vector<int> _cameras;
    cv::Size _imageSize;

    int _count;
    
    QCamCalib _camCalib;
    StereoCalib _stereoCalib;

    Controller _controller;
    Collector _collector;

};