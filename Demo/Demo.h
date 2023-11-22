#include <iostream>

#include "StereoCalib/StereoCalib.h"
#include "QCamCalib/QCamCalib.h"
#include "QNN/AIDetector/AIDetector.h"

class Demo{
public:
    enum CalibSrcType {
        TYPE_VIDEO,
        TYPE_PHOTO
    };

public:
    Demo(CalibSrcType calibSrcType, bool isOverolad = false);
    Demo(std::vector<int>& cameraId, bool isOverolad = false);
    ~Demo();

    void run();
    void run(cv::Mat imgL, cv::Mat imgR);
    void calibrate();
    void detectLandmark(const cv::Mat grayImgL, std::vector<cv::Point2f>& landmarks);

    void monoCalibate(const std::vector<std::string>& imgPathList, cv::Mat& cameraMatrix, cv::Mat& distCoeffs);

private:
    void getCalibImgs(const std::string& imgRoot, std::vector<std::string>& imgLPathList, std::vector<std::string>& imgRPathList);
    bool searchSpecifiedFiles(std::string folder, std::string extension, std::vector<std::string>& filePaths);
    void extendFaceRoi(const cv::Rect& srcRect, cv::Rect& dstRect, cv::Size size);

private:
    AIDetector* _AiDMSMTYolox{nullptr};
    AIDetector* _AiDMSMTFace{nullptr};

    StereoCalib _stereoCalib;
    QCamCalib _camCalib;

    std::vector<cv::Point2f> _landmarksL;
    std::vector<cv::Point2f> _landmarksR;

    cv::Size _cameraUnifiedSize{1280, 720};

    double _scale{0.5};
};

