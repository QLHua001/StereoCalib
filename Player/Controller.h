#pragma once

#include <list>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "ChannelObserver.h"

class Controller{
public:
    Controller(){};
    Controller(cv::Size frameSize);
    void setImgSize(cv::Size imageSize);
    virtual void attach(ChannelObserver* channel); 
    // virtual void detach(ChannelObserver* channel);
    virtual void notify();

    std::vector<cv::Mat>& getCurrFrame();
    cv::Mat& getShowMat();
    void setTitle(std::string title, double displayTime);
    void setBackgroundColor(cv::Scalar color, int thinkness);

    void release();

private:
    void addTitletoShow();
    void updateShowMat();

private:
    std::list<ChannelObserver*> _channelObserverList;
    std::vector<cv::Mat> _frames;
    cv::Mat _showMat;
    cv::Scalar _backgroudColor{0, 0, 0};
    int _thinkness{1};

    // double _delay{30}; // millisecond

    cv::Size _frameSize;
    double _scale{0.5}; // the parameter is only used to display scale

    std::string _title{""};
    double _displayTime{0};
    std::chrono::steady_clock::time_point _startTime;
    std::chrono::steady_clock::time_point _endTime;
};

