#pragma once

#include "DataType.h"

class PostProcessor{
public:
    struct Config{
        PostType type;
    };

public:
    virtual ~PostProcessor(){};
    virtual bool init(const Config* config) = 0;
    virtual void run(const NetData* netData, Output* output) = 0;

};

class PostDMSMTFace : public PostProcessor{

public:
    virtual ~PostDMSMTFace();
    virtual bool init(const Config* config);
    virtual void run(const NetData* netData, Output* output);

};

class PostDMSMTYolox : public PostProcessor{

public:
    struct Config : PostProcessor::Config{
        int classNum{-1};
        float threshold{0.0f};
        float nmsThreshold{0.4f};
    };

public:
    virtual ~PostDMSMTYolox();
    virtual bool init(const PostProcessor::Config* config);
    virtual void run(const NetData* netData, Output* output);

private:
    void nms(std::vector<Object>& objects, std::vector<float>& objScores, std::vector<Object>& nmsObjects, bool isMultiClassNMS);
    void singleClassNMS(std::vector<Object>& objects, std::vector<float>& objScores, std::vector<Object>& nmsObjects);
    void multiClassNMS(std::vector<Object>& objects, std::vector<float>& objScores,std::vector<Object>& nmsObjects);
    void qsortDescentInplace(std::vector<Object>& datas, std::vector<float>& scores);
    void qsortDescentInplace(std::vector<Object>& datas, std::vector<float>& scores, int left, int right);
    void nmsSortedBBoxes(const std::vector<Object>& bboxes, std::vector<size_t>& picked, float nmsThres);
    float intersectionArea(const Object& a, const Object& b);

private:
    Config _config;

};