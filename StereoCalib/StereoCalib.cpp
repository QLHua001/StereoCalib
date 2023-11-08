#include "StereoCalib.h"

StereoCalib::StereoCalib(){
    this->_cameraMatrixL = (cv::Mat_<double>(3, 3) << 2.5585060339489064e+03, 0., 9.6072007512833954e+02, 0.,
       2.4165383474831133e+03, 5.4057064048399172e+02, 0., 0., 1. );
    this->_distCoeffL = (cv::Mat_<double>(14, 1) << -2.0161625732351765e-01, -2.2558706856770550e+00, 0., 0., 0.,
       0., 0., 4.3526086510735666e+00, 0., 0., 0., 0., 0., 0. );

    this->_cameraMatrixR = (cv::Mat_<double>(3, 3) << 2.5585060339489064e+03, 0., 9.5576375749939166e+02, 0.,
       2.4165383474831133e+03, 5.3901319510092355e+02, 0., 0., 1. );
    this->_distCoeffR = (cv::Mat_<double>(14, 1) << -2.6721696588327870e-01, -8.3468659122720301e-01, 0., 0., 0.,
       0., 0., 1.7520486841753984e+00, 0., 0., 0., 0., 0., 0. );

    this->_R = (cv::Mat_<double>(3, 3) << 9.8899670668229323e-01, 9.7661253251007288e-02,
       -1.1112062718064832e-01, -9.6653933163027464e-02,
       9.9521345350888979e-01, 1.4429108046789707e-02,
       1.1199790790767049e-01, -3.5300946660859339e-03,
       9.9370217221054391e-01 );

    this->_T = (cv::Mat_<double>(3, 1) << 6.7222911495930981e+00, -3.5220917208478736e-01,
       2.4511341222549290e-01 );

    this->_imageSize = cv::Size(1920, 1080);

}

void StereoCalib::stereoCalibrate(){

}

void StereoCalib::stereoRectify(cv::Mat imageL, cv::Mat imageR, cv::Mat& rectifyImageL, cv::Mat& rectifyImageR){
    cv::Mat_<double> R1;
    cv::Mat_<double> R2;
    cv::Mat_<double> P1;
    cv::Mat_<double> P2;
    // cv::Mat_<double> Q;
    // cv::Rect validRoi[2];
    cv::stereoRectify(this->_cameraMatrixL, this->_distCoeffL, this->_cameraMatrixR, this->_distCoeffR,
                    this->_imageSize, this->_R, this->_T, 
                    R1, R2, P1, P2, this->_Q,
                    cv::CALIB_ZERO_DISPARITY,
                    0,
                    this->_imageSize);
    
    cv::Mat mapXL, mapYL;
    cv::Mat mapXR, mapYR;
    cv::initUndistortRectifyMap(this->_cameraMatrixL, this->_distCoeffL, R1, P1, this->_imageSize, CV_32FC1, mapXL, mapYL);
    cv::initUndistortRectifyMap(this->_cameraMatrixR, this->_distCoeffR, R2, P2, this->_imageSize, CV_32FC1, mapXR, mapYR);

    //! 经过remap之后，左右相机的图像已经共面并且行对齐
    // cv::Mat rectifyImageL, rectifyImageR;
    cv::remap(imageL, rectifyImageL, mapXL, mapYL, cv::INTER_LINEAR);
    cv::remap(imageR, rectifyImageR, mapXR, mapYR, cv::INTER_LINEAR);

    // 校验
    cv::Size showSize(this->_imageSize.width * 2, this->_imageSize.height);
    cv::Mat showRectifyImg = cv::Mat::zeros(showSize, CV_8UC1);
    rectifyImageL.copyTo(showRectifyImg(cv::Rect(0, 0, showSize.width / 2.0, showSize.height)));
    rectifyImageR.copyTo(showRectifyImg(cv::Rect(showSize.width / 2.0, 0, showSize.width / 2.0, showSize.height)));
    for(int i = 0; i < showSize.height; i += 50){
        cv::line(showRectifyImg, cv::Point(0, i), cv::Point(showSize.width-1, i), cv::Scalar(255), 1, cv::LINE_AA);
    }
    cv::imwrite("./temp/showRectifyImg.jpg", showRectifyImg);
}

void StereoCalib::stereoSGBM(const cv::Mat rectifyImageL, const cv::Mat rectifyImageR, cv::Mat& disparity){
    int minDisparity = 32;
    int numDisparities = 176;
    int blockSize = 16;
    int P1 = 4 * 1 * blockSize * blockSize;
    int P2 = 32 * 1 * blockSize * blockSize;
    int disp12MaxDiff = 1;
    int preFilterCap = 60;
    int uniquenessRatio = 30;
    int speckleWindowSize  = 200;
    int speckleRange = 2;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity, numDisparities, blockSize, P1, P2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange);

    //! rectifyImageL, rectifyImageR 8-bit single-channel image.
    // cv::Mat disparity;
    sgbm->compute(rectifyImageL, rectifyImageR, disparity);

    cv::Mat disparity8U;
    disparity.convertTo(disparity8U, CV_32F, 1/16.0);
    cv::normalize(disparity8U, disparity8U, 0, 255, cv::NormTypes::NORM_MINMAX, CV_8U);
    cv::medianBlur(disparity8U,disparity8U, 9);
    
    cv::imwrite("./temp/disparity8U.jpg", disparity8U);
    
}