
#ifdef __linux__
#include <unistd.h>
#include <dirent.h>
#elif _WIN32
#include <io.h>
#endif

#include "Demo.h"
#include "QNN/Predictor/DataTypeDMS.h"
#include "CalibSys/CalibSys.h"

Demo::Demo(CalibSrcType calibSrcType, bool isOverolad){

    this->_AiDMSMTYolox = new AIDMSMTYolox;
    this->_AiDMSMTYolox->init();

    this->_AiDMSMTFace = new AIDMSMTFace;
    this->_AiDMSMTFace->init();

    std::string calibParamYmlPath{"./config/Params.yml"};
    bool ret;
#ifdef __linux__
    ret = (access(calibParamYmlPath.c_str(), F_OK) == 0);
#elif _WIN32
    ret = (_access(calibParamYmlPath.c_str(), 0) == 0);
#endif

    if(!isOverolad && ret){

        this->_stereoCalib.loadParams(calibParamYmlPath);

    }else{
        // QCamCalib::Config camCalibConfig;
        // camCalibConfig.camType = QCamCalib::CamType::CAM_GENERAL;
        // camCalibConfig.patternSize = cv::Size(8, 5);
        // camCalibConfig.squareSize = cv::Size(30, 30);
        // camCalibConfig.srcImgSize = cv::Size(1920 * this->_scale, 1080 * this->_scale);
        // if(!this->_camCalib.init(camCalibConfig)){
        //     printf("QCamCalib init fail!\n");
        // }
        // printf("QCamCalib init successfully!\n");

        // this->calibrate();

        //########################################################################################

        // std::vector<int> cameras{0, 1};
        // CalibSys calibSys(CalibSys::InputType::TYPE_CAMERA, cameras, cv::Size(1920, 1080));

        std::vector<std::string> videos{
            "./example/calib_8/videoL.mp4",
            "./example/calib_8/videoR.mp4"
        };
        CalibSys calibSys(CalibSys::InputType::TYPE_VIDEO, videos, cv::Size(1920, 1080));

        calibSys.init();
        bool ret = calibSys.run();
        if(ret == true){
            this->_stereoCalib.loadParams(calibParamYmlPath);
        }
    }

}

Demo::Demo(std::vector<int>& cameraId, bool isOverolad){
    this->_AiDMSMTYolox = new AIDMSMTYolox;
    this->_AiDMSMTYolox->init();

    this->_AiDMSMTFace = new AIDMSMTFace;
    this->_AiDMSMTFace->init();

    std::string calibParamYmlPath{"./config/Params.yml"};
    bool ret;
#ifdef __linux__
    ret = (access(calibParamYmlPath.c_str(), F_OK) == 0);
#elif _WIN32
    ret = (_access(calibParamYmlPath.c_str(), 0) == 0);
#endif

    if(!isOverolad && ret){

        this->_stereoCalib.loadParams(calibParamYmlPath);

    }else{
        // QCamCalib::Config camCalibConfig;
        // camCalibConfig.camType = QCamCalib::CamType::CAM_GENERAL;
        // camCalibConfig.patternSize = cv::Size(8, 5);
        // camCalibConfig.squareSize = cv::Size(30, 30);
        // camCalibConfig.srcImgSize = cv::Size(1920 * this->_scale, 1080 * this->_scale);
        // if(!this->_camCalib.init(camCalibConfig)){
        //     printf("QCamCalib init fail!\n");
        // }
        // printf("QCamCalib init successfully!\n");

        // this->calibrate();

        //########################################################################################

        // std::vector<int> cameras{0, 1};
        CalibSys calibSys(CalibSys::InputType::TYPE_CAMERA, cameraId, this->_cameraUnifiedSize);

        // std::vector<std::string> videos{
        //     "./example/calib_8/videoL.mp4",
        //     "./example/calib_8/videoR.mp4"
        // };
        // CalibSys calibSys(CalibSys::InputType::TYPE_VIDEO, videos, cv::Size(1920, 1080));

        calibSys.init();
        bool ret = calibSys.run();
        if(ret == true){
            this->_stereoCalib.loadParams(calibParamYmlPath);
        }
    }
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

static double getDistance(const cv::Point3f& p1, const cv::Point3f& p2){
    double xd = p1.x - p2.x;
    double yd = p1.y - p2.y;
    double zd = p1.z - p2.z;
    double result = std::sqrt(xd * xd + yd * yd + zd * zd);
    return result;
}

void Demo::run(){

    std::string videoL{"./example/calib_imgs_5_video/outputL1.mp4"};
    std::string videoR{"./example/calib_imgs_5_video/outputR1.mp4"};
    std::string outputPath{""};

    cv::VideoCapture capL(videoL);
    if(!capL.isOpened()){
        printf("open video fail!(%s)\n", videoL.c_str());
        return;
    }

    cv::VideoCapture capR(videoR);
    if(!capR.isOpened()){
        printf("open video fail!(%s)\n", videoR.c_str());
        return;
    }

    int frameW = capL.get(cv::CAP_PROP_FRAME_WIDTH) * this->_scale;
    int frameH = capL.get(cv::CAP_PROP_FRAME_HEIGHT) * this->_scale;
    int fps = int(capL.get(cv::CAP_PROP_FPS));
    printf("frameW: %d\n", frameW);
    printf("frameH: %d\n", frameH);
    printf("fps: %d\n", fps);

    cv::VideoWriter writer;
    if(!outputPath.empty()){
        writer.open(outputPath.c_str(), cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 15, cv::Size(frameW, frameH*1.5));
        if(!writer.isOpened()){
            printf("VideoWriter open fail!\n");
            return;
        }
    }

    cv::Mat imgL;
    cv::Mat imgR;
    while(true){

        capL >> imgL;
        capR >> imgR;

        // imgL = testImgL;
        // imgR = testImgR;
        if(imgL.empty() || imgR.empty()) break;

        cv::resize(imgL, imgL, cv::Size(), this->_scale, this->_scale);
        cv::resize(imgR, imgR, cv::Size(), this->_scale, this->_scale);

        cv::Mat grayImgL, grayImgR;
        cv::cvtColor(imgL, grayImgL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, grayImgR, cv::COLOR_BGR2GRAY);

        cv::Mat rectifyImageL;
        cv::Mat rectifyImageR;
        this->_stereoCalib.stereoRectify(grayImgL, grayImgR, rectifyImageL, rectifyImageR);
        // cv::imwrite("./temp/rectifyImageL.jpg", rectifyImageL);
        // cv::imwrite("./temp/rectifyImageR.jpg", rectifyImageR);

        cv::Mat showMat(imgL.rows * 3 / 2, imgL.cols, CV_8UC3, cv::Scalar(0, 0, 0));
        imgL.copyTo(showMat(cv::Rect(0, 0, imgL.cols, imgL.rows)));
        int y_offset = imgL.rows + 30;

        this->detectLandmark(rectifyImageL, this->_landmarksL);
        this->detectLandmark(rectifyImageR, this->_landmarksR);
        if(!(this->_landmarksL.empty()) || !(this->_landmarksR.empty())) {

            cv::Mat filteredDisparityColorMap;
            cv::Mat xyz; //! CV_32FC3
            std::vector<cv::Point3f> worldPts;
            this->_stereoCalib.stereoSGBM(rectifyImageL, rectifyImageR, this->_landmarksL, this->_landmarksR, filteredDisparityColorMap, xyz, worldPts);
            // cv::imwrite("./temp/filteredDisparityColorMap.jpg", filteredDisparityColorMap);
            // std::cout << "xyz channels(): " << xyz.channels() << std::endl;
            // std::cout << "xyz type(): " << xyz.type() << std::endl;

            // cv::Mat showMat1 = rectifyImageL.clone();
            // cv::cvtColor(showMat1, showMat1, cv::COLOR_GRAY2BGR);
            // std::vector<cv::Point2f> imagePts;
            // this->_stereoCalib.projectPoint(worldPts, imagePts);
            // for(const auto& pt : imagePts){
            //     cv::circle(showMat1, pt, 5, cv::Scalar(0, 0, 255), 1);
            // }
            // cv::imwrite("./temp/showMat1.jpg", showMat1);

            // filteredDisparityColorMap.copyTo(showMat(cv::Rect(0, 0, imgL.cols, imgL.rows)));

            // eye distance
            double eyeDistance = getDistance(worldPts[0], worldPts[1]);
            double mouthDistance = getDistance(worldPts[3], worldPts[4]);

            std::vector<cv::Point2f> srcLandmarkL;
            this->detectLandmark(grayImgL, srcLandmarkL);
            if(srcLandmarkL.empty()) {
                printf("srcLandmarkL is empty.\n");
                continue;
            }

            for(int i = 0; i < srcLandmarkL.size(); i++){
                cv::circle(showMat, srcLandmarkL[i], 3, cv::Scalar(0, 255, 0), -1);
            }

            char printStr[64];
            // std::vector<int> index{6, 10, 14, 15, 17};
            // for(int i = 0; i < index.size(); i++){
            //     int t = index[i];
            //     cv::Point2f srcPt = srcLandmarkL[t];
            //     cv::Point3f wPt = worldPts[i];
            //     sprintf(printStr, "x:%.2f", wPt.x);
            //     cv::putText(imgL, printStr, cv::Point(srcPt.x-5, srcPt.y-15), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
            //     sprintf(printStr, "y:%.2f", wPt.y);
            //     cv::putText(imgL, printStr, cv::Point(srcPt.x-5, srcPt.y-10), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
            //     sprintf(printStr, "z:%.2f", wPt.z);
            //     cv::putText(imgL, printStr, cv::Point(srcPt.x-5, srcPt.y-5), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
            // }

            sprintf(printStr, "###left eye depth: %.2f", worldPts[0].z);
            cv::putText(showMat, printStr, cv::Point(10, 0 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            sprintf(printStr, "##right eye depth: %.2f", worldPts[1].z);
            cv::putText(showMat, printStr, cv::Point(10, 30 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            sprintf(printStr, "######nose depth: %.2f", worldPts[2].z);
            cv::putText(showMat, printStr, cv::Point(10, 60 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            sprintf(printStr, "#left mouth depth: %.2f", worldPts[3].z);
            cv::putText(showMat, printStr, cv::Point(10, 90 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            sprintf(printStr, "right mouth depth: %.2f", worldPts[4].z);
            cv::putText(showMat, printStr, cv::Point(10, 120 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // std::sort(worldPts.begin(), worldPts.end(), [](const cv::Point3f& p1, const cv::Point3f& p2){
            //     return p1.z < p2.z;
            // });

            double face_depth = ((worldPts[0].z + worldPts[1].z) / 2.0) - worldPts[2].z;

            // double face_max_depth = worldPts[worldPts.size()-1].z - worldPts[0].z;
            sprintf(printStr, "######face depth: %.2f", face_depth);
            cv::putText(showMat, printStr, cv::Point(10, 150 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            cv::Point2f p1, p2;

            p1 = srcLandmarkL[6];
            p2 = srcLandmarkL[10];
            cv::line(showMat, p1, p2, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            sprintf(printStr, "distance: %.2f", eyeDistance);
            cv::putText(showMat, printStr, cv::Point((p1.x+p2.x)/2, (p1.y+p2.y)/2-5), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

            p1 = srcLandmarkL[15];
            p2 = srcLandmarkL[17];
            cv::line(showMat, p1, p2, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            sprintf(printStr, "distance: %.2f", mouthDistance);
            cv::putText(showMat, printStr, cv::Point((p1.x+p2.x)/2, (p1.y+p2.y)/2-5), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

            // cv::imwrite("./temp/showMat.jpg", showMat);
        }
    
#ifdef _WIN32
        // cv::Mat showImg;
        // cv::resize(imgL, showImg, cv::Size(imgL.cols / 2.0, imgL.rows / 2.0));
        // cv::imshow("showMat", showMat);
        // cv::waitKey(10);
#endif

        if(!outputPath.empty()){
            writer.write(showMat);
        }
        // //! print depth
        // for(int i = 0; i < this->_landmarksL.size(); i++){
        //     cv::Point2f ptL = this->_landmarksL[i];
        //     float x = xyz.at<cv::Vec3f>(int(ptL.y), int(ptL.x))[0];
        //     float y = xyz.at<cv::Vec3f>(int(ptL.y), int(ptL.x))[1];
        //     float z = xyz.at<cv::Vec3f>(int(ptL.y), int(ptL.x))[2];
        //     std::cout << i << " [" << ptL.x << ", " << ptL.y << "]: " << x << ", " << y << ", " << z << std::endl;

        //     // cv::Point2f ptR = this->_landmarksR[i];
        //     // this->_stereoCalib.calcStereoDist(ptL.x, ptL.y, ptR.x, ptR.y);
        // }

        // for(int i = 0; i < this->_landmarksL.size(); i++){
        //     cv::Point2f ptL = this->_landmarksL[i];
        //     cv::Point2f ptR = this->_landmarksR[i];
        //     printf("index: %d --- left: %.2f, %.2f, right: %.2f, %.2f\n", i, ptL.x, ptL.y, ptR.x, ptR.y);
        // }
    }
    if(!outputPath.empty()){
        writer.release();
    }
    capL.release();
    capR.release();
}

void Demo::run(cv::Mat imgL, cv::Mat imgR){

    if((imgL.cols != this->_cameraUnifiedSize.width) || (imgL.rows != this->_cameraUnifiedSize.height)){
        cv::resize(imgL, imgL, this->_cameraUnifiedSize);
    }
    if((imgR.cols != this->_cameraUnifiedSize.width) || (imgR.rows != this->_cameraUnifiedSize.height)){
        cv::resize(imgR, imgR, this->_cameraUnifiedSize);
    }

    cv::Mat grayImgL, grayImgR;
    cv::cvtColor(imgL, grayImgL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgR, grayImgR, cv::COLOR_BGR2GRAY);

    cv::Mat rectifyImageL;
    cv::Mat rectifyImageR;
    this->_stereoCalib.stereoRectify(grayImgL, grayImgR, rectifyImageL, rectifyImageR);
    // cv::imwrite("./temp/rectifyImageL.jpg", rectifyImageL);
    // cv::imwrite("./temp/rectifyImageR.jpg", rectifyImageR);
    cv::Mat showMat(imgL.rows * 3 / 2, imgL.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    imgL.copyTo(showMat(cv::Rect(0, 0, imgL.cols, imgL.rows)));
    int y_offset = imgL.rows + 30;

    this->detectLandmark(rectifyImageL, this->_landmarksL);
    this->detectLandmark(rectifyImageR, this->_landmarksR);
    if(!(this->_landmarksL.empty()) && !(this->_landmarksR.empty())) {

        cv::Mat filteredDisparityColorMap;
        cv::Mat xyz; //! CV_32FC3
        std::vector<cv::Point3f> worldPts;
        this->_stereoCalib.stereoSGBM(rectifyImageL, rectifyImageR, this->_landmarksL, this->_landmarksR, filteredDisparityColorMap, xyz, worldPts);
        // cv::imwrite("./temp/filteredDisparityColorMap.jpg", filteredDisparityColorMap);
        // std::cout << "xyz channels(): " << xyz.channels() << std::endl;
        // std::cout << "xyz type(): " << xyz.type() << std::endl;

        // cv::Mat showMat1 = rectifyImageL.clone();
        // cv::cvtColor(showMat1, showMat1, cv::COLOR_GRAY2BGR);
        // std::vector<cv::Point2f> imagePts;
        // this->_stereoCalib.projectPoint(worldPts, imagePts);
        // for(const auto& pt : imagePts){
        //     cv::circle(showMat1, pt, 5, cv::Scalar(0, 0, 255), 1);
        // }
        // cv::imwrite("./temp/showMat1.jpg", showMat1);

        // filteredDisparityColorMap.copyTo(showMat(cv::Rect(0, 0, imgL.cols, imgL.rows)));

        // eye distance
        double eyeDistance = getDistance(worldPts[0], worldPts[1]);
        double mouthDistance = getDistance(worldPts[3], worldPts[4]);

        std::vector<cv::Point2f> srcLandmarkL;
        this->detectLandmark(grayImgL, srcLandmarkL);
        if(srcLandmarkL.empty()) {
            printf("srcLandmarkL is empty.\n");
            return;
        }

        for(int i = 0; i < srcLandmarkL.size(); i++){
            cv::circle(showMat, srcLandmarkL[i], 3, cv::Scalar(0, 255, 0), -1);
        }

        char printStr[64];
        // std::vector<int> index{6, 10, 14, 15, 17};
        // for(int i = 0; i < index.size(); i++){
        //     int t = index[i];
        //     cv::Point2f srcPt = srcLandmarkL[t];
        //     cv::Point3f wPt = worldPts[i];
        //     sprintf(printStr, "x:%.2f", wPt.x);
        //     cv::putText(imgL, printStr, cv::Point(srcPt.x-5, srcPt.y-15), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
        //     sprintf(printStr, "y:%.2f", wPt.y);
        //     cv::putText(imgL, printStr, cv::Point(srcPt.x-5, srcPt.y-10), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
        //     sprintf(printStr, "z:%.2f", wPt.z);
        //     cv::putText(imgL, printStr, cv::Point(srcPt.x-5, srcPt.y-5), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));
        // }

        sprintf(printStr, "###left eye depth: %.2f", worldPts[0].z);
        cv::putText(showMat, printStr, cv::Point(10, 0 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        sprintf(printStr, "##right eye depth: %.2f", worldPts[1].z);
        cv::putText(showMat, printStr, cv::Point(10, 30 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        sprintf(printStr, "######nose depth: %.2f", worldPts[2].z);
        cv::putText(showMat, printStr, cv::Point(10, 60 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        sprintf(printStr, "#left mouth depth: %.2f", worldPts[3].z);
        cv::putText(showMat, printStr, cv::Point(10, 90 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        sprintf(printStr, "right mouth depth: %.2f", worldPts[4].z);
        cv::putText(showMat, printStr, cv::Point(10, 120 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 255, 0), 2);

        // std::sort(worldPts.begin(), worldPts.end(), [](const cv::Point3f& p1, const cv::Point3f& p2){
        //     return p1.z < p2.z;
        // });

        double face_depth = ((worldPts[0].z + worldPts[1].z) / 2.0) - worldPts[2].z;

        // double face_max_depth = worldPts[worldPts.size()-1].z - worldPts[0].z;
        sprintf(printStr, "######face depth: %.2f", face_depth);
        cv::putText(showMat, printStr, cv::Point(10, 150 + y_offset), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2);

        cv::Point2f p1, p2;

        p1 = srcLandmarkL[6];
        p2 = srcLandmarkL[10];
        cv::line(showMat, p1, p2, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        sprintf(printStr, "distance: %.2f", eyeDistance);
        cv::putText(showMat, printStr, cv::Point((p1.x+p2.x)/2, (p1.y+p2.y)/2-5), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        p1 = srcLandmarkL[15];
        p2 = srcLandmarkL[17];
        cv::line(showMat, p1, p2, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        sprintf(printStr, "distance: %.2f", mouthDistance);
        cv::putText(showMat, printStr, cv::Point((p1.x+p2.x)/2, (p1.y+p2.y)/2-5), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // cv::imwrite("./temp/showMat.jpg", showMat);
    }

#ifdef _WIN32
        cv::resize(showMat, showMat, cv::Size(), 0.5, 0.5);
        cv::imshow("showMat", showMat);
        cv::waitKey(10);
#endif

}

void Demo::calibrate(){

    std::string imgRoot{"./example/calib_imgs_6/"};

    std::vector<std::string> imgLPathList, imgRPathList;
    this->getCalibImgs(imgRoot, imgLPathList, imgRPathList);

    // left camera
    cv::Mat cameraMatrixL;
    cv::Mat distCoeffsL;
    cv::Mat cameraMatrixR;
    cv::Mat distCoeffsR;

    this->monoCalibate(imgLPathList, cameraMatrixL, distCoeffsL);
    this->_stereoCalib.setIntrisicsL(cameraMatrixL, distCoeffsL);

    this->monoCalibate(imgRPathList, cameraMatrixR, distCoeffsR);
    this->_stereoCalib.setIntrisicsR(cameraMatrixR, distCoeffsR);

    this->_stereoCalib.stereoCalibrate(imgLPathList, imgRPathList);

}

void Demo::detectLandmark(const cv::Mat grayImgL, std::vector<cv::Point2f>& landmarks){

    landmarks.clear();

    cv::Mat gray888;
    cv::cvtColor(grayImgL, gray888, cv::COLOR_GRAY2BGR);
    // cv::imwrite("./temp/gray888.jpg", gray888);

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
                // cv::circle(showImg, pt, 2, cv::Scalar(255, 255, 0), 1);
                // cv::putText(showImg, std::to_string(i).c_str(), cv::Point(pt.x, pt.y-5), cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0), 1);
            }
            break;
        }
        }
    }
    // cv::imwrite("./temp/showImg.jpg", showImg);
}

void Demo::monoCalibate(const std::vector<std::string>& imgPathList, cv::Mat& cameraMatrix, cv::Mat& distCoeffs){

    std::vector<std::vector<cv::Point2f>> imagePoints;
    for(int i = 0; i < imgPathList.size(); i++){
        std::string filePath = imgPathList[i];
        cv::Mat srcImg = cv::imread(filePath);
        if(srcImg.empty()){
            printf("imread %s fail!\n", filePath.c_str());
            continue;
        }
        cv::resize(srcImg, srcImg, cv::Size(), this->_scale, this->_scale);

        std::vector<cv::Point2f> corners;
        if(!this->_camCalib.findChessboardCorners(srcImg, corners)){
            printf("findChessboardCorners fail!(%s)\n", filePath.c_str());
            continue;
        }
        // cv::Mat showImg = srcImg.clone();
        // cv::drawChessboardCorners(showImg, patternSize, corners, true);
        // cv::imwrite("./temp/showImg.jpg", showImg);
        // usleep(3*1000*1000);
        imagePoints.push_back(corners);
    }

    this->_camCalib.calibrateCamera(imagePoints, cameraMatrix, distCoeffs);
    std::cout << "monoCalibate output- cameraMatrix: \n" << cameraMatrix << std::endl;
    std::cout << "monoCalibate output- distCoeffs: \n" << distCoeffs << std::endl;

}

void Demo::getCalibImgs(const std::string& imgRoot, std::vector<std::string>& imgLPathList, std::vector<std::string>& imgRPathList){
    std::vector<std::string> filePaths;
    searchSpecifiedFiles(imgRoot, ".jpg", filePaths);
    printf("Found %ld imgs\n", filePaths.size());

    int imgNum = filePaths.size();
    for(int i = 0; i < imgNum/2; i++){
        char basename[128];

        sprintf(basename, "left_%d.jpg", i);
        imgLPathList.push_back(imgRoot + basename);

        sprintf(basename, "right_%d.jpg", i);
        imgRPathList.push_back(imgRoot + basename);
    }
}

bool Demo::searchSpecifiedFiles(std::string folder, std::string extension, std::vector<std::string>& filePaths){
    if(folder.empty() || extension.empty()) return false; 

    if(folder.back() != '/'){
        folder += "/";
    }

#ifdef _WIN32

    std::string searchPath = folder + "/*" + extension;
    intptr_t handle;
    _finddata_t fileInfo;

    // 查找第一个文件/目录
    handle = _findfirst(searchPath.c_str(), &fileInfo);
    if (handle == -1)
    {
        std::cout << "Failed to find first file" << std::endl;
        return false;
    }

    do
    {
        if (!(fileInfo.attrib & _A_SUBDIR))
        {
            std::string filename(fileInfo.name);
            std::string filePath = folder + "/" + filename;
            //std::cout << "File: " << filePath << std::endl;
            filePaths.push_back(filePath);
        }
    } while (_findnext(handle, &fileInfo) == 0);

    _findclose(handle);

#elif __linux__

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
#endif

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