#include <iostream>
#include "Player.h"

Player::Player(std::string videoPath){
    this->_videoPath = videoPath;
    this->_cap = cv::VideoCapture(this->_videoPath);
    if(!this->_cap.isOpened()){
        printf("cv::VideoCapture open fail!(%s)\n", this->_videoPath.c_str());
        exit(-1);
    }
    printf("cv::VideoCapture open successfully!(%s)\n", this->_videoPath.c_str());
}

Player::Player(int deviceId){
    this->_deviceId = deviceId;
    this->_cap = cv::VideoCapture(this->_deviceId);
    if(!this->_cap.isOpened()){
        printf("cv::VideoCapture open fail!(id: %d)\n", this->_deviceId);
        exit(-1);
    }
    printf("cv::VideoCapture open successfully!(id: %d)\n", this->_deviceId);
}

Player::~Player(){
    printf("Player Release.\n");
    this->_cap.release();
}

int Player::snapshot(cv::Mat& frame){
    std::cout << this->_videoPath << " snapshot..." << std::endl;

    this->_cap.read(frame);

    // for convenience to test, scale the image to unified size 
    if((frame.cols != 1920) || (frame.rows != 1080)){
        cv::resize(frame, frame, cv::Size(1920, 1080));
    }

    return 0;
}