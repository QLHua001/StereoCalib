#ifdef __linux__
#include <unistd.h>
#endif

#include <chrono>

#include <opencv2/opencv.hpp>
#include "Controller.h"

Controller::Controller(cv::Size frameSize){
    this->_frameSize = frameSize;
}

void Controller::setImgSize(cv::Size imageSize){
    this->_frameSize = imageSize;
}

void Controller::attach(ChannelObserver* channel){
    this->_channelObserverList.push_back(channel);
    this->_frames.push_back(cv::Mat(this->_frameSize, CV_8UC3, cv::Scalar(0, 0, 0)));
    this->updateShowMat();
}

// void Controller::detach(ChannelObserver* channel){
//     this->_channelObserverList.remove(channel);
//     this->updateShowMat();
// }

void Controller::notify(){
    std::list<ChannelObserver*>::iterator it = this->_channelObserverList.begin();
    int index = 0;
    while(it != this->_channelObserverList.end()){
        cv::Mat frame;
        (*it)->snapshot(frame);
        if(!frame.empty()){
            this->_frames[index] = frame.clone();
            cv::resize(frame, frame, cv::Size(), this->_scale, this->_scale);
            frame.copyTo(this->_showMat(cv::Rect(this->_frameSize.width*this->_scale*index, 0, this->_frameSize.width*this->_scale, this->_frameSize.height*this->_scale)));
        }
        ++it;
        ++index;
    }

    cv::rectangle(this->_showMat, cv::Rect(0, 0, this->_showMat.cols, this->_showMat.rows), this->_backgroudColor, this->_thinkness);

    this->addTitletoShow();
}

std::vector<cv::Mat>& Controller::getCurrFrame(){
    return this->_frames;
}

cv::Mat& Controller::getShowMat(){
    return this->_showMat;
}

void Controller::setTitle(std::string title, double displayTime){
    this->_startTime = std::chrono::steady_clock::now();
    this->_displayTime = displayTime;
    this->_title = title;
}

void Controller::setBackgroundColor(cv::Scalar color, int thinkness){
    this->_backgroudColor = color;
    this->_thinkness = thinkness;
}

void Controller::release(){
    std::list<ChannelObserver*>::iterator it = this->_channelObserverList.begin();
    while(it != this->_channelObserverList.end()){
        delete (*it);
        ++it;
    }
}

void Controller::addTitletoShow(){
    if(!this->_title.empty()){
        this->_endTime = std::chrono::steady_clock::now();
        if(std::chrono::duration_cast<std::chrono::seconds>(this->_endTime-this->_startTime).count() < this->_displayTime){
            int fontFace = cv::FONT_HERSHEY_SIMPLEX;
            double fontScale = 2.5;
            int thickness = 2;
            cv::Size textSize = cv::getTextSize(this->_title, fontFace, fontScale, thickness, nullptr);
            cv::Point textOrigin((this->_showMat.cols - textSize.width) / 2.0, (this->_showMat.rows + textSize.height) / 2.0);
            cv::putText(this->_showMat, this->_title, textOrigin, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
        }else{
            this->_title = "";
            this->_displayTime = 0;
        }
    }
}

void Controller::updateShowMat(){
    cv::Size displaySize = this->_frameSize;
    displaySize.width *= this->_scale;
    displaySize.height *= this->_scale;

    this->_showMat = cv::Mat(displaySize.height, displaySize.width * this->_channelObserverList.size(), CV_8UC3, cv::Scalar(0, 0, 0));
}

