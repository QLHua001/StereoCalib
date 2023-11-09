#pragma once 

#include "DataType.h"

class PreProcessor{
public:
    struct Config{
        int tarWidth;
        int tarHeight;
        ImgFormat tarFormat;

        Config& operator=(const Config& config){
            this->tarWidth = config.tarWidth;
            this->tarHeight = config.tarHeight;
            this->tarFormat = config.tarFormat;
            return *this;
        }
    };

public:
    ~PreProcessor();

    bool init(const Config* config);
    void run(const Input* input, Tensor* tensor);

private:
    Config _config;
    unsigned char* _cache;

};