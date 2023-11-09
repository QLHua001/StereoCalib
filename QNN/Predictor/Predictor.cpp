#include "Predictor.h"

Predictor::~Predictor(){
    if(this->_preProcessor){
        delete this->_preProcessor;
    }
    this->_preProcessor = nullptr;

    if(this->_net){
        delete this->_net;
    }
    this->_net = nullptr;

    if(this->_postProcessor){
        delete this->_postProcessor;
    }
    this->_postProcessor = nullptr;
}

bool Predictor::init(const Config* config){

    this->_preProcessor = new PreProcessor;
    this->_preProcessor->init(config->preConfig);

    switch(config->netConfig->type){
    case NetType::NET_NCNN:{
        this->_net = new NetNCNN;
        this->_net->init(config->netConfig);
        break;
    }
    }

    switch(config->postConfig->type){
    case PostType::POST_DMS_MTFACE:{
        this->_postProcessor = new PostDMSMTFace;
        this->_postProcessor->init(config->postConfig);
        break;
    }
    case PostType::POST_DMS_MTYOLOX:{
        this->_postProcessor = new PostDMSMTYolox;
        this->_postProcessor->init(config->postConfig);
        break;
    }
    }

    return true;
}

 void Predictor::run(const Input* input, Output* output){
    
    printf("Predictor run..\n");

    this->_preProcessor->run(input, &this->_tensor);

    this->_net->run(&this->_tensor, &this->_netData);

    this->_postProcessor->run(&this->_netData, output);

 }