#pragma once

#include <opencv2/opencv.hpp>

class StereoCalib{

public:
    StereoCalib(cv::Size imageSize = cv::Size(1920, 1080), double scale = 1.0, cv::Size patternSize = cv::Size(8, 5), cv::Size squareSize = cv::Size(30, 30));

    bool loadParams(const std::string& calibParamYmlPath);
    void setIntrisicsL(cv::Mat cameraMatrix, cv::Mat distCoeffs);
    void setIntrisicsR(cv::Mat cameraMatrix, cv::Mat distCoeffs);

    void stereoCalibrate(const std::vector<std::string>& imgLPathList, const std::vector<std::string>& imgRPathList);
    void stereoCalibrate(std::vector<std::vector<cv::Point2f>>& imagePointsL, std::vector<std::vector<cv::Point2f>>& imagePointsR);

    void stereoRectify(const cv::Mat imageL, const cv::Mat imageR, cv::Mat& rectifyImageL, cv::Mat& rectifyImageR);
    void stereoSGBM(const cv::Mat rectifyImageL, const cv::Mat rectifyImageR, std::vector<cv::Point2f>& landmarkL, std::vector<cv::Point2f>& landmarkR, cv::Mat& filteredDisparityColorMap, cv::Mat& xyz, std::vector<cv::Point3f>& worldPts);
    // void projectPoint(const std::vector<cv::Point3f>& worldPts, std::vector<cv::Point2f>& imagePts);

    void saveParams(const std::string& path);
    
    void calcStereoDist(double lx, double ly, double rx, double ry);
    cv::Point3f calculateDistance(cv::Point3f in);

private:
    cv::Mat_<double> _cameraMatrixL;
    cv::Mat_<double> _distCoeffL;
    cv::Mat_<double> _cameraMatrixR;
    cv::Mat_<double> _distCoeffR;
    cv::Mat_<double> _R;
    cv::Mat_<double> _T;
    cv::Mat_<double> _E;
    cv::Mat_<double> _F;
    cv::Size _imageSize;

    cv::Mat_<double> _Q;
    
    cv::Rect _validRoi[2];
    cv::Mat _mapXL;
    cv::Mat _mapYL;
    cv::Mat _mapXR;
    cv::Mat _mapYR;

    double _scale{1.0};
    cv::Size _patternSize;
    cv::Size _squareSize;

    cv::Ptr<cv::StereoSGBM> _sgbm;
};