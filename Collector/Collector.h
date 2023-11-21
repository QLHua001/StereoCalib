#pragma once 

#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>

#include "QCamCalib/QCamCalib.h"

class Collector{

public:
    Collector(){};
    Collector(size_t count);

    void setCamCalibConfig(const QCamCalib::Config& config);

    void startCollect();

    int inputImage(std::vector<cv::Mat>& frames);

    std::vector<std::vector<std::vector<cv::Point2f>>>& getCorners();

private:
    int findChessboardCorners(std::vector<cv::Mat>& frames);

private:
    bool _startFlag{false};
    int _cornerCount;

    std::chrono::steady_clock::time_point _startTime;
    std::chrono::steady_clock::time_point _currTime;

    double _timeInterval1{5}; // seconds
    double _timeInterval2{7}; // seconds
    double _timeInterval3{9}; // seconds

    int _sampleNum{3};
    int _count;

    QCamCalib _camCalib;
    std::vector<std::vector<std::vector<cv::Point2f>>> _corners;
};