#include "Collector.h"

Collector::Collector(size_t count){

    // QCamCalib::Config camCalibConfig;
    // camCalibConfig.camType = QCamCalib::CamType::CAM_GENERAL;
    // camCalibConfig.patternSize = cv::Size(8, 5);
    // camCalibConfig.squareSize = cv::Size(30, 30);
    // camCalibConfig.srcImgSize = cv::Size(1920, 1080);
    // if(!this->_camCalib.init(camCalibConfig)){
    //     printf("QCamCalib init fail!\n");
    // }
    // printf("QCamCalib init successfully!\n");
    this->_cornerCount = count;
    this->_corners.resize(this->_cornerCount);
}

void Collector::setCamCalibConfig(const QCamCalib::Config& config){
    if(!this->_camCalib.init(config)){
        printf("QCamCalib init fail!\n");
    }
    printf("QCamCalib init successfully!\n");
}

void Collector::startCollect(){
    this->_startFlag = true;
    this->_count = 0;
    this->_corners.clear();
    this->_corners.resize(this->_cornerCount);
    this->_startTime = std::chrono::steady_clock::now();
}

int Collector::inputImage(std::vector<cv::Mat>& frames){
    if(!this->_startFlag) return 4;

    this->_currTime = std::chrono::steady_clock::now();
    double interval = std::chrono::duration_cast<std::chrono::seconds>(this->_currTime - this->_startTime).count();
    if(interval < this->_timeInterval1){
        return 1;
    }else if((this->_timeInterval1 <= interval) && (interval < this->_timeInterval2)){
        return 2;
    }else if((this->_timeInterval2 <= interval) && (interval < this->_timeInterval3)){
        return 3;
    }else{
        // collect;
        int ret = findChessboardCorners(frames);
        if(ret == frames.size()) {
            this->_count++;
        }else{
            printf("%d not found corners\n", ret);
        }
        printf("sample count: %d\n", this->_count);
        if(this->_count == this->_sampleNum){
            printf("Collect %d Images successfully!\n", this->_count);
            this->_startFlag = false;
            this->_count = 0;
            return 5; // collect finish.
        }

        this->_startTime = std::chrono::steady_clock::now();

        return 4;
    }
}

std::vector<std::vector<std::vector<cv::Point2f>>>& Collector::getCorners(){
    return this->_corners;
}

int Collector::findChessboardCorners(std::vector<cv::Mat>& frames){

    std::vector<std::vector<cv::Point2f>> temp;
    for(int i = 0; i < frames.size(); i++){
        std::vector<cv::Point2f> corners;
        bool isFound = this->_camCalib.findChessboardCorners(frames[i], corners);
        if(!isFound) return i;
        temp.push_back(corners);
    }

    for(int i = 0; i < temp.size(); i++){
        this->_corners[i].push_back(temp[i]);
    }

    return temp.size();
}