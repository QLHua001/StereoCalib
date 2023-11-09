#include "PostProcessor.h"
#include "DataTypeDMS.h"

PostDMSMTYolox::~PostDMSMTYolox(){

}

bool PostDMSMTYolox::init(const PostProcessor::Config* config){
    this->_config = *(Config*)config;
    return true;
}

void PostDMSMTYolox::run(const NetData* netData, Output* output){

    printf("PostDMSMTYolox run...\n");
    std::vector<float> strides{8.0f, 16.0f, 32.0f, 64.0f};

    DMSMTYolox* pOutput = (DMSMTYolox*)output;

    std::vector<Object> objects;
    std::vector<float> objScores;
    for(int i = 0; i < strides.size(); i++){
        float* pBboxBranch = netData->data[i*3+0];
        float* pObjBranch = netData->data[i*3+1];
        float* pClsBranch = netData->data[i*3+2];

        int gridW = netData->shapes[i*3+0].w;
        int gridH = netData->shapes[i*3+0].h;
        for(int y = 0; y < gridH; y++){
            for(int x = 0; x < gridW; x++){
                int p = y * gridW + x;
                float objScore = pObjBranch[p];
                if(objScore < this->_config.threshold) continue;
                
                for(int c = 0; c < this->_config.classNum; c++){
                    float clsScore = pClsBranch[p+c*gridW*gridH];
                    clsScore *= objScore;
                    if(clsScore < this->_config.threshold) continue;

                    float centerX = (pBboxBranch[p+0*gridW*gridH] + x) * strides[i];
                    float centerY = (pBboxBranch[p+1*gridW*gridH] + y) * strides[i];
                    int w = std::exp(pBboxBranch[p+2*gridW*gridH]) * strides[i];
                    int h = std::exp(pBboxBranch[p+3*gridW*gridH]) * strides[i];
                    float xmin = centerX - w * 0.5f;
                    float ymin = centerY - h * 0.5f;

                    Object obj;
                    obj.bbox = cv::Rect(xmin, ymin, w, h);
                    obj.label = c;
                    obj.score = clsScore;
                    objects.push_back(obj);
                    objScores.push_back(clsScore);
                }
            }
        }
    }

    // nms
    std::vector<Object> nmsObjects;
    this->nms(objects, objScores, nmsObjects, false);

    for(auto& obj : nmsObjects){
        obj.bbox.x *= netData->sx;
        obj.bbox.y *= netData->sy;
        obj.bbox.width *= netData->sx;
        obj.bbox.height *= netData->sy;
        pOutput->objects.push_back(obj);
    }

    float* pData = netData->data[12];
    pOutput->attributes.push_back(Attribute{"FLAG_NO_BELT", pData[0]});
    pOutput->attributes.push_back(Attribute{"FLAG_OCCLUSIVE", pData[1]});
    pOutput->attributes.push_back(Attribute{"FLAG_FACE", pData[2]});
    pOutput->attributes.push_back(Attribute{"FLAG_CALL", pData[3]});
    pOutput->attributes.push_back(Attribute{"FLAG_SMOKE", pData[4]});
}

void PostDMSMTYolox::nms(std::vector<Object>& objects, std::vector<float>& objScores, std::vector<Object>& nmsObjects, bool isMultiClassNMS){
    if(isMultiClassNMS){
        this->multiClassNMS(objects, objScores, nmsObjects);
    }else{
        this->singleClassNMS(objects, objScores, nmsObjects);
    }
}

void PostDMSMTYolox::singleClassNMS(std::vector<Object>& objects, std::vector<float>& objScores, std::vector<Object>& nmsObjects){
    nmsObjects.clear();

    this->qsortDescentInplace(objects, objScores);

    std::map<int, std::vector<Object>> objMap;
    std::map<int, std::vector<float>> objScoreMap;
    for(int i = 0; i < objects.size(); i++){
        Object obj = objects[i];
        objMap[obj.label].push_back(obj);
        objScoreMap[obj.label].push_back(objScores[i]);
    }

    for(const auto& item : objMap){
        const auto& label = item.first;
        const auto& objects = item.second;

        // nms
        std::vector<size_t> picked;
        this->nmsSortedBBoxes(objects, picked, this->_config.nmsThreshold);

        int count = picked.size();
        for(int i = 0; i < count; i++){
            nmsObjects.push_back(objects[picked[i]]);
        }
    }
}

void PostDMSMTYolox::multiClassNMS(std::vector<Object>& objects, std::vector<float>& objScores,std::vector<Object>& nmsObjects){

}

void PostDMSMTYolox::qsortDescentInplace(std::vector<Object>& datas, std::vector<float>& scores){
    if (datas.empty() || scores.empty())
        return;
    this->qsortDescentInplace(datas, scores, 0, static_cast<int>(scores.size() - 1));
}

void PostDMSMTYolox::qsortDescentInplace(std::vector<Object>& datas, std::vector<float>& scores, int left, int right){
    int   i = left;
    int   j = right;
    float p = scores[(left + right) / 2];

    while (i <= j) {
        while (scores[i] > p)
            i++;

        while (scores[j] < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(datas[i], datas[j]);
            std::swap(scores[i], scores[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        this->qsortDescentInplace(datas, scores, left, j);

    if (i < right)
        this->qsortDescentInplace(datas, scores, i, right);
}

void PostDMSMTYolox::nmsSortedBBoxes(const std::vector<Object>& bboxes, std::vector<size_t>& picked, float nmsThres){
    picked.clear();

    const size_t n = bboxes.size();

    std::vector<float> areas(n);
    for (size_t i = 0; i < n; i++) {
        const Object &r = bboxes[i];

        float width  = r.bbox.br().x - r.bbox.tl().x;
        float height = r.bbox.br().y - r.bbox.tl().y;

        areas[i] = width * height;
    }

    for (size_t i = 0; i < n; i++) {
        const Object &a = bboxes[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const Object &b = bboxes[picked[j]];

            // intersection over union
            float inter_area = this->intersectionArea(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nmsThres)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

float PostDMSMTYolox::intersectionArea(const Object& a, const Object& b){
    if(a.bbox.tl().x > b.bbox.br().x || a.bbox.br().x < b.bbox.tl().x || a.bbox.tl().y > b.bbox.br().y || a.bbox.br().y < b.bbox.tl().y){
        return 0.0f;
    }

    float inter_width  = std::min(a.bbox.br().x, b.bbox.br().x) - std::max(a.bbox.tl().x, b.bbox.tl().x);
    float inter_height = std::min(a.bbox.br().y, b.bbox.br().y) - std::max(a.bbox.tl().y, b.bbox.tl().y);

    return inter_width * inter_height;
}