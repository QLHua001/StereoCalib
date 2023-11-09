#include "AIDetector.h"

#include "./example/models/ncnn/DMSMTFace/DFMTN_16f.param.h"
#include "./example/models/ncnn/DMSMTFace/DFMTN_16f.bin.h"

AIDMSMTFace::~AIDMSMTFace(){
    if(this->_predictor){
        delete this->_predictor;
    }
    this->_predictor = nullptr;
}

void AIDMSMTFace::init(){

    PreProcessor::Config preConfig;
    preConfig.tarWidth = 160;
    preConfig.tarHeight = 160;
    preConfig.tarFormat = ImgFormat::FMT_BGR888;

    // Net::Config netConfig;
    NetNCNN::Config netConfig;
    netConfig.type = NetType::NET_NCNN;
    netConfig.mean = std::array<float, 3>{127.5, 127.5, 127.5};
    netConfig.std = std::array<float, 3>{0.0078125, 0.0078125, 0.0078125};
    netConfig.pModelParam = DFMTN_16f_param_bin;
    netConfig.pModelWeight = DFMTN_16f_bin;
    netConfig.inputNodes = std::vector<int>{DFMTN_16f_param_id::BLOB_input};
    netConfig.outputNodes = std::vector<int>{DFMTN_16f_param_id::BLOB_lds, DFMTN_16f_param_id::BLOB_euler, DFMTN_16f_param_id::BLOB_cls1, DFMTN_16f_param_id::BLOB_cls2};
    netConfig.outputShapes = std::vector<NCHW>{
        NCHW{1, 1, 1, 40},
        NCHW{1, 1, 1, 3},
        NCHW{1, 1, 1, 4},
        NCHW{1, 1, 1, 4}
    };

    PostProcessor::Config postConfig;
    postConfig.type = PostType::POST_DMS_MTFACE;

    Predictor::Config config;
    config.preConfig = &preConfig;
    config.netConfig = &netConfig;
    config.postConfig = &postConfig;

    this->_predictor = new Predictor;
    if(!this->_predictor->init(&config)){
        printf("AIDMSMTFace::Predictor init fail!\n");
        return;
    }
    printf("AIDMSMTFace::Predictor init successfully!\n");



}

void AIDMSMTFace::run(const Input* input, Output* output){

    this->_predictor->run(input, output);

}