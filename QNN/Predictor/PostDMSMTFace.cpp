#include "PostProcessor.h"
#include "DataTypeDMS.h"

PostDMSMTFace::~PostDMSMTFace(){

}

bool PostDMSMTFace::init(const Config* config){

    return true;
}

void PostDMSMTFace::run(const NetData* netData, Output* output){
    
    printf("PostDMSMTFace run...\n");

    DMSMTFace* pOutput = (DMSMTFace*)output;

    // landmark
    pOutput->landmark.type = LandmarkType::BSJ_20;
    for(int i = 0; i < netData->shapes[0].size; i+=2){
        cv::Point2f pt(netData->data[0][i], netData->data[0][i+1]);
        pOutput->landmark.points.push_back(pt);
    }

    // euler
    double pitch = netData->data[1][0] * 180 / M_PI;
    double yaw = netData->data[1][1] * 180 / M_PI;
    double roll = netData->data[1][2] * 180 / M_PI;
    // printf("pitch: %.2f, yaw: %.2f, roll: %.2f\n", pitch, yaw, roll);
    pOutput->eulerAngle.pitch = pitch;
    pOutput->eulerAngle.yaw = yaw;
    pOutput->eulerAngle.roll = roll;

    pOutput->attributes.push_back(Attribute{"FLAG_SMOKE", netData->data[2][0]});
    pOutput->attributes.push_back(Attribute{"FLAG_YAWN", netData->data[2][1]});
    pOutput->attributes.push_back(Attribute{"FLAG_MASK", netData->data[2][2]});
    pOutput->attributes.push_back(Attribute{"FLAG_IR_REJECTION", netData->data[3][0]});
    pOutput->attributes.push_back(Attribute{"FLAG_LEFTEYE_INVISIBLE", netData->data[3][1]});
    pOutput->attributes.push_back(Attribute{"FLAG_RIGHTEYE_INVISIBLE", netData->data[3][2]});
    pOutput->attributes.push_back(Attribute{"FLAG_MOUTH_INVISIBLE", netData->data[3][3]});
    // printf("smoke: %.2f, yawn: %.2f, mask: %.2f\n", netData->data[2][0], netData->data[2][1], netData->data[2][2]);
    // printf("ir: %.2f, lefteye_invisible: %.2f, righteye_invisible: %.2f, mouth_invisible: %.2f\n", netData->data[3][0], netData->data[3][1], netData->data[3][2], netData->data[3][3]);
}