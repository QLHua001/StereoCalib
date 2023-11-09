#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>

#include "DataType.h"
#include "PreProcessor.h"
#include "Net.h"
#include "PostProcessor.h"

class Predictor{
public:
    struct Config{
        PreProcessor::Config* preConfig;
        Net::Config* netConfig;
        PostProcessor::Config* postConfig;
    };

public:
    ~Predictor();

    bool init(const Config* config);
    void run(const Input* input, Output* output);

private:
    PreProcessor* _preProcessor{nullptr};
    Net* _net{nullptr};
    PostProcessor* _postProcessor{nullptr};

    Tensor _tensor;
    NetData _netData;
};