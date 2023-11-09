#pragma once

#include "QNN/Predictor/Predictor.h"

class AIDetector{
public:
    virtual ~AIDetector(){}
    virtual void init() = 0;
    virtual void run(const Input* input, Output* output) = 0;
};

class AIDMSMTFace : public AIDetector{
public:
    virtual ~AIDMSMTFace();
    virtual void init();
    virtual void run(const Input* input, Output* output);

private:
    Predictor* _predictor{nullptr};
};

class AIDMSMTYolox : public AIDetector{
public:
    virtual ~AIDMSMTYolox();
    virtual void init();
    virtual void run(const Input* input, Output* output);

private:
    Predictor* _predictor{nullptr};
};