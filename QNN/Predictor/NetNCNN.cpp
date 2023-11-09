#include <ncnn/net.h>
#include "Net.h"

NetNCNN::~NetNCNN(){
    if(this->_net){
        delete (ncnn::Net*)this->_net;
    }
    this->_net = nullptr;

    for(int i = 0; i < this->_cache.size(); i++){
        if(this->_cache[i]){
            delete [] this->_cache[i];
        }
        this->_cache[i] = nullptr;
    }
}

bool NetNCNN::init(const Net::Config* config){
    this->_config = *(Config*)config;
    for(const auto& val : this->_config.mean){
        printf("%f\n", val);
    }

    ncnn::Net* net = new ncnn::Net;
    if((net->load_param(this->_config.pModelParam) == 0) || (net->load_model(this->_config.pModelWeight) == 0)){
        printf("ncnn::Net load fail!\n");
        return false;
    }
    this->_net = (void*)net;

    this->_cache.resize(this->_config.outputShapes.size());
    for(int i = 0; i < this->_config.outputShapes.size(); i++){
        this->_cache[i] = new float[this->_config.outputShapes[i].size];
        printf("this->_cache[%d]: %p\n", i, this->_cache[i]);
    }

    return true;
}

void NetNCNN::run(const Tensor* tensor, NetData* netData){

    printf("NetNCNN run...\n");

    ncnn::Mat input = ncnn::Mat::from_pixels(tensor->data, ncnn::Mat::PIXEL_BGR, tensor->width, tensor->height);
    input.substract_mean_normalize(this->_config.mean.data(), this->_config.std.data());

    ncnn::Extractor ex = ((ncnn::Net*)this->_net)->create_extractor();
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input(this->_config.inputNodes[0], input);

    std::vector<ncnn::Mat> outputs(this->_config.outputNodes.size());
    for(int i = 0; i < this->_config.outputNodes.size(); i++){
        ex.extract(this->_config.outputNodes[i], outputs[i]);
        printf("d: %d, c: %d, h: %d, w: %d\n", outputs[i].d, outputs[i].c, outputs[i].h, outputs[i].w);

        float* pData = (float*)(outputs[i].data);
        memcpy(this->_cache[i], pData, this->_config.outputShapes[i].size*sizeof(float));
        // int dataLen = outputs[i].d * outputs[i].c * outputs[i].h * outputs[i].w;
        // for(int j = 0; j < dataLen; j++){
        //     netData->data[i].push_back(pData[j]);
        // }
    }

    netData->clear();
    for(int i = 0; i < this->_cache.size(); i++){
        netData->data.push_back(this->_cache[i]);
        netData->shapes.push_back(this->_config.outputShapes[i]);
    }
    netData->sx = tensor->sx;
    netData->sy = tensor->sy;
}