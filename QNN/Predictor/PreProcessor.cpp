#include <iostream>
#include "PreProcessor.h"

PreProcessor::~PreProcessor(){
    if(this->_cache){
        delete [] this->_cache;
    }
    this->_cache = nullptr;
}

bool PreProcessor::init(const Config* config){

    printf("PreProcessor init...\n");

    this->_config = *config;

    switch(this->_config.tarFormat){
    case ImgFormat::FMT_BGR888:{
        this->_cache = new unsigned char[this->_config.tarWidth * this->_config.tarHeight * 3];
        break;
    }
    }

    return true;
}

void PreProcessor::run(const Input* input, Tensor* tensor){
    
    printf("PreProcessor run...\n");

    cv::Mat srcImg = cv::Mat(input->height, input->width, CV_8UC3, input->data);
    
    cv::Mat dstImg;
    switch(this->_config.tarFormat){
    case ImgFormat::FMT_BGR888:{
        dstImg = cv::Mat(this->_config.tarHeight, this->_config.tarWidth, CV_8UC3, this->_cache);
        break;
    }
    }
    cv::resize(srcImg, dstImg, cv::Size(this->_config.tarWidth, this->_config.tarHeight));

    tensor->data = this->_cache;
    tensor->width = this->_config.tarWidth;
    tensor->height = this->_config.tarHeight;
    tensor->format = this->_config.tarFormat;
    tensor->sx = double(input->width) / double(this->_config.tarWidth);
    tensor->sy = double(input->height) / double(this->_config.tarHeight);

}