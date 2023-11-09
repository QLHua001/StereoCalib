#pragma once

#include <array>
#include <vector>
#include "DataType.h"

class Net{
public:
    struct Config{
        NetType type;
        std::array<float, 3> mean;
        std::array<float, 3> std;

        // Config& operator=(const Config& config){
        //     printf("Net::Config operator=\n");
        //     this->type = config.type;
        //     this->mean = config.mean;
        //     this->std = std;
        //     return *this;
        // }
    };

public:
    virtual ~Net(){};
    virtual bool init(const Config* config) = 0;
    virtual void run(const Tensor* tensor, NetData* netData) = 0;

};

class NetNCNN : public Net{
public:
    struct Config : Net::Config{
        const unsigned char* pModelParam;
        const unsigned char* pModelWeight;
        std::vector<int> inputNodes;
        std::vector<int> outputNodes;
        std::vector<NCHW> outputShapes;

        // Config& operator=(const Config& config){
        //     printf("NetNCNN::Config operator=\n");
        //     // *(Net::Config*)this = config;
        //     this->pModelParam = config.pModelParam;
        //     this->pModelWeight = config.pModelWeight;
        //     this->inputNodes.clear();
        //     this->inputNodes.insert(this->inputNodes.end(), config.inputNodes.begin(), config.inputNodes.end());
        //     this->outputNodes.clear();
        //     this->outputNodes.insert(this->outputNodes.end(), config.outputNodes.begin(), config.outputNodes.end());
        //     return *this;
        // }
    };

public:
    virtual ~NetNCNN();
    bool init(const Net::Config* config);
    void run(const Tensor* tensor, NetData* netData);

private:
    Config _config;
    void* _net{nullptr};

    std::vector<float*> _cache;
};