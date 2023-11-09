#include "AIDetector.h"

#include "./example/models/ncnn/DMSMTYolox/Yolox_16f.param.h"
#include "./example/models/ncnn/DMSMTYolox/Yolox_16f.bin.h"

AIDMSMTYolox::~AIDMSMTYolox(){
    if(this->_predictor){
        delete this->_predictor;
    }
    this->_predictor = nullptr;
}

void AIDMSMTYolox::init(){
    PreProcessor::Config preConfig;
    preConfig.tarWidth = 416;
    preConfig.tarHeight = 256;
    preConfig.tarFormat = ImgFormat::FMT_BGR888;

    NetNCNN::Config netConfig;
    netConfig.type = NetType::NET_NCNN;
    netConfig.mean = std::array<float, 3>{127.5, 127.5, 127.5};
    netConfig.std = std::array<float, 3>{0.0078125, 0.0078125, 0.0078125};
    netConfig.pModelParam = Yolox_16f_param_bin;
    netConfig.pModelWeight = Yolox_16f_bin;
    netConfig.inputNodes = std::vector<int>{Yolox_16f_param_id::BLOB_input};
    netConfig.outputNodes = std::vector<int>{Yolox_16f_param_id::BLOB_bbox8, Yolox_16f_param_id::BLOB_obj8, Yolox_16f_param_id::BLOB_cls8,
                                            Yolox_16f_param_id::BLOB_bbox16, Yolox_16f_param_id::BLOB_obj16, Yolox_16f_param_id::BLOB_cls16,
                                            Yolox_16f_param_id::BLOB_bbox32, Yolox_16f_param_id::BLOB_obj32, Yolox_16f_param_id::BLOB_cls32,
                                            Yolox_16f_param_id::BLOB_bbox64, Yolox_16f_param_id::BLOB_obj64, Yolox_16f_param_id::BLOB_cls64,
                                            Yolox_16f_param_id::BLOB_output_1};
    netConfig.outputShapes = std::vector<NCHW>{
        NCHW{1, 4, 32, 52}, NCHW{1, 1, 32, 52}, NCHW{1, 8, 32, 52},
        NCHW{1, 4, 16, 26}, NCHW{1, 1, 16, 26}, NCHW{1, 8, 16, 26},
        NCHW{1, 4, 8, 13},  NCHW{1, 1, 8, 13},  NCHW{1, 8, 8, 13},
        NCHW{1, 4, 4, 7},   NCHW{1, 1, 4, 7},   NCHW{1, 8, 4, 7},
        NCHW{1, 1, 1, 5}
    };

    // PostProcessor::Config postConfig;
    // postConfig.type = PostType::POST_DMS_MTYOLOX;
    PostDMSMTYolox::Config postConfig;
    postConfig.type = PostType::POST_DMS_MTYOLOX;
    postConfig.classNum = 8;
    postConfig.threshold = 0.4f;
    postConfig.nmsThreshold = 0.4f;


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

void AIDMSMTYolox::run(const Input* input, Output* output){
    this->_predictor->run(input, output);
}
