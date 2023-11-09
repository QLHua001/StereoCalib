#include <dirent.h>

#include "Demo.h"
#include "QNN/Predictor/DataTypeDMS.h"

Demo::Demo(){

    this->_AiDMSMTYolox = new AIDMSMTYolox;
    this->_AiDMSMTYolox->init();

    this->_AiDMSMTFace = new AIDMSMTFace;
    this->_AiDMSMTFace->init();

    this->calibrate();

}

Demo::~Demo(){
    if(this->_AiDMSMTYolox){
        delete this->_AiDMSMTYolox;
    }
    this->_AiDMSMTYolox = nullptr;

    if(this->_AiDMSMTFace){
        delete this->_AiDMSMTFace;
    }
    this->_AiDMSMTFace = nullptr;
}

void Demo::run(){
    std::string imgLPath{"./example/calib_imgs_3/left38.jpg"};
    std::string imgRPath{"./example/calib_imgs_3/left38.jpg"};

    cv::Mat imgL = cv::imread(imgLPath);
    if(imgL.empty()){
        printf("imread fail!(%s)\n", imgLPath.c_str());
        return;
    }

    cv::Mat imgR = cv::imread(imgRPath);
    if(imgR.empty()){
        printf("imread fail!(%s)\n", imgRPath.c_str());
        return;
    }

    cv::Mat grayImgL, grayImgR;
    cv::cvtColor(imgL, grayImgL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, grayImgR, cv::COLOR_BGR2GRAY);

    this->detectLandmark(grayImgL, this->_landmarksL);
    this->detectLandmark(grayImgR, this->_landmarksR);

    cv::Mat rectifyImageL;
    cv::Mat rectifyImageR;
    this->_stereoCalib.stereoRectify(grayImgL, grayImgR, rectifyImageL, rectifyImageR);
    cv::imwrite("./temp/rectifyImageL.jpg", rectifyImageL);
    cv::imwrite("./temp/rectifyImageR.jpg", rectifyImageR);

    cv::Mat filteredDisparityColorMap;
    cv::Mat xyz; //! CV_32FC3
    this->_stereoCalib.stereoSGBM(rectifyImageL, rectifyImageR, filteredDisparityColorMap, xyz);
    cv::imwrite("./temp/filteredDisparityColorMap.jpg", filteredDisparityColorMap);
    std::cout << "xyz channels(): " << xyz.channels() << std::endl;
    std::cout << "xyz type(): " << xyz.type() << std::endl;

    //! print depth
    for(int i = 0; i < this->_landmarksL.size(); i++){
        cv::Point2f ptL = this->_landmarksL[i];
        float x = xyz.at<cv::Vec3f>(int(ptL.y), int(ptL.x))[0];
        float y = xyz.at<cv::Vec3f>(int(ptL.y), int(ptL.x))[1];
        float z = xyz.at<cv::Vec3f>(int(ptL.y), int(ptL.x))[2];
        std::cout << i << " [" << ptL.x << ", " << ptL.y << "]: " << x << ", " << y << ", " << z << std::endl;

        cv::Point2f ptR = this->_landmarksR[i];
        this->_stereoCalib.calcStereoDist(ptL.x, ptL.y, ptR.x, ptR.y);
    }

    for(int i = 0; i < this->_landmarksL.size(); i++){
        cv::Point2f ptL = this->_landmarksL[i];
        cv::Point2f ptR = this->_landmarksR[i];
        printf("index: %d --- left: %.2f, %.2f, right: %.2f, %.2f\n", i, ptL.x, ptL.y, ptR.x, ptR.y);
    }

}

void Demo::calibrate(){

    std::string imgRoot{"./example/calib_imgs_3/"};

    std::vector<std::string> imgLPathList, imgRPathList;
    this->getCalibImgs(imgRoot, imgLPathList, imgRPathList);

    this->_stereoCalib.stereoCalibrate(imgLPathList, imgRPathList);

}

void Demo::detectLandmark(const cv::Mat grayImgL, std::vector<cv::Point2f>& landmarks){
    cv::Mat gray888;
    cv::cvtColor(grayImgL, gray888, cv::COLOR_GRAY2BGR);
    cv::imwrite("./temp/gray888.jpg", gray888);

    cv::Mat inputImg = gray888;

    Input DMSMTYoloxInput;
    DMSMTYoloxInput.data = inputImg.data;
    DMSMTYoloxInput.width = inputImg.cols;
    DMSMTYoloxInput.height = inputImg.rows;
    DMSMTYoloxInput.format = ImgFormat::FMT_BGR888;
    DMSMTYolox DMSMTYoloxOutput;
    this->_AiDMSMTYolox->run(&DMSMTYoloxInput, &DMSMTYoloxOutput);

    cv::Mat showImg = inputImg.clone();
    for(int i = 0; i < DMSMTYoloxOutput.objects.size(); i++){
        Object& obj = DMSMTYoloxOutput.objects[i];
        if(obj.label != 0) continue; // 不是人脸，跳过。

        cv::Rect faceRoi = obj.bbox;
        cv::Rect faceRoiEx;
        this->extendFaceRoi(faceRoi, faceRoiEx, cv::Size(inputImg.cols, inputImg.rows));
        cv::Mat faceImgEx = inputImg(faceRoiEx);
        if(!faceImgEx.isContinuous()) faceImgEx = faceImgEx.clone();

        Input DMSMTFaceInput;
        DMSMTFaceInput.data = faceImgEx.data;
        DMSMTFaceInput.width = faceImgEx.cols;
        DMSMTFaceInput.height = faceImgEx.rows;
        DMSMTFaceInput.format = ImgFormat::FMT_BGR888;
        DMSMTFace DMSMTFaceOutput;
        this->_AiDMSMTFace->run(&DMSMTFaceInput, &DMSMTFaceOutput);

        cv::rectangle(showImg, faceRoi, cv::Scalar(0, 255, 0), 1);
        LandmarkType landmarkType = DMSMTFaceOutput.landmark.type;
        switch(landmarkType){
        case LandmarkType::BSJ_20:{
            for(int i = 0; i < 20; i++){
                cv::Point2d pt = DMSMTFaceOutput.landmark.points[i];
                pt.x *= faceImgEx.cols;
                pt.y *= faceImgEx.rows;
                pt.x += faceRoiEx.x;
                pt.y += faceRoiEx.y;
                landmarks.push_back(pt);
                cv::circle(showImg, pt, 2, cv::Scalar(255, 255, 0), 1);
                cv::putText(showImg, std::to_string(i).c_str(), cv::Point(pt.x, pt.y-5), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0), 1);
            }
            break;
        }
        }
    }
    cv::imwrite("./temp/showImg.jpg", showImg);
}

void Demo::getCalibImgs(const std::string& imgRoot, std::vector<std::string>& imgLPathList, std::vector<std::string>& imgRPathList){
    std::vector<std::string> filePaths;
    searchSpecifiedFiles(imgRoot, ".jpg", filePaths);
    printf("Found %ld imgs\n", filePaths.size());

    int imgNum = filePaths.size();
    for(int i = 0; i < imgNum/2; i++){
        char basename[128];

        sprintf(basename, "left%d.jpg", i);
        imgLPathList.push_back(imgRoot + basename);

        sprintf(basename, "right%d.jpg", i);
        imgRPathList.push_back(imgRoot + basename);
    }
}

bool Demo::searchSpecifiedFiles(std::string folder, std::string extension, std::vector<std::string>& filePaths){
    if(folder.empty() || extension.empty()) return false; 

    if(folder.back() != '/'){
        folder += "/";
    }

    DIR* pDir;
    dirent* pCur;

    if((pDir = opendir(folder.c_str())) == NULL){
        printf("opendir %s fail!\n", folder.c_str());
        return false;
    }

    while((pCur = readdir(pDir)) != NULL){
        if(DT_REG == pCur->d_type){
            const char* selfExtension = strrchr(pCur->d_name, '.');
            if(selfExtension != NULL && strcmp(selfExtension, extension.c_str()) == 0){
                std::string filePath = folder + pCur->d_name;
                filePaths.push_back(filePath);
            }
        }
    }

    return true;
}

void Demo::extendFaceRoi(const cv::Rect& srcRect, cv::Rect& dstRect, cv::Size size){
    int exW = srcRect.width * 0.1;
    int exH = srcRect.height * 0.1;

    int x1 = srcRect.tl().x - exW;
    x1 = (x1 < 0) ? 0 : x1;

    int y1 = srcRect.tl().y - exH;
    y1 = (y1 < 0) ? 0 : y1;

    int x2 = srcRect.br().x + exW;
    x2 = (x2 >= size.width) ? size.width - 1 : x2;

    int y2 = srcRect.br().y + exH;
    y2 = (y2 >= size.height) ? size.height - 1 : y2;

    dstRect.x = x1;
    dstRect.y = y1;
    dstRect.width = x2 - x1 + 1;
    dstRect.height = y2 - y1 + 1;
}