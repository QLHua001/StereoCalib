#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>

enum ImgFormat{
    FMT_BGR888,
    FMT_NV12
};

enum NetType{
    NET_NCNN,
    NET_RKNN
};

struct NCHW {
    int n;
    int c;
    int h;
    int w;
    int size;

    NCHW(int n = 0, int c = 0, int h = 0, int w = 0) {
        this->n = n;
        this->c = c;
        this->h = h;
        this->w = w;
        this->size = this->n * this->c * this->h * this->w;
    }
};

enum PostType{
    POST_DMS_MTFACE,
    POST_DMS_MTYOLOX
};

struct Attribute{
    std::string label;
    double score;
};

struct Object{
    cv::Rect bbox;
    int label;
    double score;
};

struct Input{
    unsigned char* data;
    int width;
    int height;
    ImgFormat format;
};

struct Tensor{
    unsigned char* data;
    int width;
    int height;
    ImgFormat format;
    double sx;
    double sy;
};

struct NetData{
    std::vector<float*> data;
    std::vector<NCHW> shapes;
    double sx;
    double sy;

    void clear(){
        this->data.clear();
        this->shapes.clear();
    }
};

struct Output{

};