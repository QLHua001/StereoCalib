#include <iostream>

#include "StereoCalib/StereoCalib.h"
#include "QNN/AIDetector/AIDetector.h"

class Demo{
public:
    Demo();
    ~Demo();

    void run();
    void calibrate();
    void detectLandmark(const cv::Mat grayImgL, std::vector<cv::Point2f>& landmarks);

private:
    void getCalibImgs(const std::string& imgRoot, std::vector<std::string>& imgLPathList, std::vector<std::string>& imgRPathList);
    bool searchSpecifiedFiles(std::string folder, std::string extension, std::vector<std::string>& filePaths);
    void extendFaceRoi(const cv::Rect& srcRect, cv::Rect& dstRect, cv::Size size);

private:
    AIDetector* _AiDMSMTYolox{nullptr};
    AIDetector* _AiDMSMTFace{nullptr};

    StereoCalib _stereoCalib;

    std::vector<cv::Point2f> _landmarksL;
    std::vector<cv::Point2f> _landmarksR;
};

