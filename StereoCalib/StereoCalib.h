#pragma once

#include <opencv2/opencv.hpp>

class StereoCalib{

public:
    StereoCalib();

    void stereoCalibrate();
    void stereoRectify(const cv::Mat imageL, const cv::Mat imageR, cv::Mat& rectifyImageL, cv::Mat& rectifyImageR);
    void stereoSGBM(const cv::Mat rectifyImageL, const cv::Mat rectifyImageR, cv::Mat& disparity);

private:
    cv::Mat_<double> _cameraMatrixL;
    cv::Mat_<double> _distCoeffL;
    cv::Mat_<double> _cameraMatrixR;
    cv::Mat_<double> _distCoeffR;
    cv::Mat_<double> _R;
    cv::Mat_<double> _T;
    cv::Size _imageSize;

    cv::Mat_<double> _Q;

};