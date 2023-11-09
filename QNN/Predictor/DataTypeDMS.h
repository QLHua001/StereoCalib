#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>

#include "DataType.h"

struct DMSMTYolox : Output{
    std::vector<Object> objects;
    std::vector<Attribute> attributes;
};

enum LandmarkType{
    NONE,
    BSJ_20,
    DLIB_68
};

struct Landmark{
    LandmarkType type;
    std::vector<cv::Point2f> points;

    Landmark(){
        this->type = LandmarkType::NONE;
    }

    Landmark& operator=(const Landmark& landmark){
        this->type = landmark.type;
        this->points.clear();
        this->points.insert(this->points.end(), landmark.points.begin(), landmark.points.end());
        return *this;
    }
};

struct EulerAngle{
    double pitch;
    double yaw;
    double roll;

    EulerAngle(double pitch = 0, double yaw = 0, double roll = 0){
        this->pitch = pitch;
        this->yaw = yaw;
        this->roll = roll;
    }

    EulerAngle& operator=(const EulerAngle& eulerAngle){
        this->pitch = eulerAngle.pitch;
        this->yaw = eulerAngle.yaw;
        this->roll = eulerAngle.roll;
        return *this;
    }

};

// enum Label{
//     FLAG_NORMAL,
//     FLAG_SMOKE,
//     FLAG_YAWN,
//     FLAG_MASK,
//     FLAG_IR_REJECTION,
//     FLAG_LEFTEYE_INVISIBLE,
//     FLAG_RIGHTEYE_INVISIBLE,
//     FLAG_MOUTH_INVISIBLE
// };

struct DMSMTFace : Output{
    Landmark landmark;
    EulerAngle eulerAngle;
    std::vector<Attribute> attributes;
};