#include <opencv2/ximgproc.hpp>
#include "StereoCalib.h"

StereoCalib::StereoCalib(){

    std::cout << "new StereoCalib" << std::endl;

    // this->_cameraMatrixL = (cv::Mat_<double>(3, 3) << 2.5585060339489064e+03, 0., 9.6072007512833954e+02, 0.,
    //    2.4165383474831133e+03, 5.4057064048399172e+02, 0., 0., 1. );
    // this->_distCoeffL = (cv::Mat_<double>(14, 1) << -2.0161625732351765e-01, -2.2558706856770550e+00, 0., 0., 0.,
    //    0., 0., 4.3526086510735666e+00, 0., 0., 0., 0., 0., 0. );

    // this->_cameraMatrixR = (cv::Mat_<double>(3, 3) << 2.5585060339489064e+03, 0., 9.5576375749939166e+02, 0.,
    //    2.4165383474831133e+03, 5.3901319510092355e+02, 0., 0., 1. );
    // this->_distCoeffR = (cv::Mat_<double>(14, 1) << -2.6721696588327870e-01, -8.3468659122720301e-01, 0., 0., 0.,
    //    0., 0., 1.7520486841753984e+00, 0., 0., 0., 0., 0., 0. );

    // this->_R = (cv::Mat_<double>(3, 3) << 9.8899670668229323e-01, 9.7661253251007288e-02,
    //    -1.1112062718064832e-01, -9.6653933163027464e-02,
    //    9.9521345350888979e-01, 1.4429108046789707e-02,
    //    1.1199790790767049e-01, -3.5300946660859339e-03,
    //    9.9370217221054391e-01 );

    // this->_T = (cv::Mat_<double>(3, 1) << 6.7222911495930981e+00, -3.5220917208478736e-01,
    //    2.4511341222549290e-01 );

    this->_imageSize = cv::Size(1920, 1080);

}

void StereoCalib::setIntrisicsL(cv::Mat cameraMatrix, cv::Mat distCoeffs){
    this->_cameraMatrixL = cameraMatrix.clone();
    this->_distCoeffL = distCoeffs.clone();

    std::cout << "setIntrisicsL output- cameraMatrix: \n" << this->_cameraMatrixL << std::endl;
    std::cout << "setIntrisicsL output- distCoeffs: \n" << this->_distCoeffL << std::endl;
}

void StereoCalib::setIntrisicsR(cv::Mat cameraMatrix, cv::Mat distCoeffs){
    this->_cameraMatrixR = cameraMatrix.clone();
    this->_distCoeffR = distCoeffs.clone();

    std::cout << "setIntrisicsR output- cameraMatrix: \n" << this->_cameraMatrixR << std::endl;
    std::cout << "setIntrisicsR output- distCoeffs: \n" << this->_distCoeffR << std::endl;
}

void StereoCalib::stereoCalibrate(const std::vector<std::string>& imgLPathList, const std::vector<std::string>& imgRPathList){

    cv::Size patternSize(8, 5);
    cv::Size squareSize(30, 30); // mm 

    std::vector<std::vector<cv::Point2f>> imagePointsL;
    std::vector<std::vector<cv::Point2f>> imagePointsR;
    std::vector<std::vector<cv::Point3f>> objectPoints;

    int imgNum = imgLPathList.size();
    int success = 0;
    for(int i = 0; i < imgNum; i++){
        std::string imgLPath = imgLPathList[i];
        std::string imgRPath = imgRPathList[i];

        cv::Mat imgL = cv::imread(imgLPath);
        cv::Mat imgR = cv::imread(imgRPath);

        cv::Mat grayImgL, grayImgR;
        cv::cvtColor(imgL, grayImgL, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, grayImgR, cv::COLOR_BGR2GRAY);

        bool foundL, foundR;
        std::vector<cv::Point2f> cornerL, cornerR;
        foundL = cv::findChessboardCorners(grayImgL, patternSize, cornerL, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        foundR = cv::findChessboardCorners(grayImgR, patternSize, cornerR, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
        if(foundL && foundR){
            cv::cornerSubPix(grayImgL, cornerL, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            cv::cornerSubPix(grayImgR, cornerR, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
            imagePointsL.push_back(cornerL);
            imagePointsR.push_back(cornerR);

            success += 1;

            std::cout << "The image  " << i << "  is good" << std::endl;

            // draw
            cv::Mat showImgL_chessboard, showImgR_chessboard;
            showImgL_chessboard = imgL.clone();
            showImgR_chessboard = imgR.clone();
            cv::drawChessboardCorners(showImgL_chessboard, patternSize, cornerL, foundL);
            cv::drawChessboardCorners(showImgR_chessboard, patternSize, cornerR, foundR);
            cv::imwrite("./temp/showImgL_chessboard.jpg", showImgL_chessboard);
            cv::imwrite("./temp/showImgR_chessboard.jpg", showImgR_chessboard);
        }else{
            std::cout << "The image is bad please try again" << std::endl;
        }
    }
    
    // objectPoints
    for(int i = 0; i < success; i++){
        std::vector<cv::Point3f> corners;
        for(int row = 0; row < patternSize.height; row++){
            for(int col = 0; col < patternSize.width; col++){
                corners.push_back(cv::Point3f(col * squareSize.width, row * squareSize.height, 0.0f));
            }
        }
        objectPoints.push_back(corners);
    }

    std::cout << "Start stereoCalibrate..." << std::endl;

    double err = cv::stereoCalibrate(objectPoints, imagePointsL, imagePointsR,
                                    this->_cameraMatrixL, this->_distCoeffL,
                                    this->_cameraMatrixR, this->_distCoeffR,
                                    this->_imageSize,
                                    this->_R, this->_T, this->_E, this->_F,
                                    cv::CALIB_USE_INTRINSIC_GUESS,
                                    cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));
    std::cout << "ML: " << std::endl << this->_cameraMatrixL << std::endl;
    std::cout << "DL: " << std::endl << this->_distCoeffL << std::endl;
    std::cout << "MR: " << std::endl << this->_cameraMatrixR << std::endl;
    std::cout << "DR: " << std::endl << this->_distCoeffR << std::endl;
    std::cout << "R: " << std::endl << this->_R << std::endl;
    std::cout << "T: " << std::endl << this->_T << std::endl;


    std::cout << "The err = " << err << std::endl;

    cv::Mat_<double> R1;
    cv::Mat_<double> R2;
    cv::Mat_<double> P1;
    cv::Mat_<double> P2;
    // cv::Mat_<double> Q;
    
    cv::stereoRectify(this->_cameraMatrixL, this->_distCoeffL, this->_cameraMatrixR, this->_distCoeffR,
                    this->_imageSize, this->_R, this->_T, 
                    R1, R2, P1, P2, this->_Q,
                    cv::CALIB_ZERO_DISPARITY,
                    -1,
                    this->_imageSize, &this->_validRoi[0], &this->_validRoi[1]);
    std::cout << "R1: " << std::endl << R1 << std::endl;
    std::cout << "R2: " << std::endl << R2 << std::endl;
    std::cout << "P1: " << std::endl << P1 << std::endl;
    std::cout << "P2: " << std::endl << P2 << std::endl;
    std::cout << "Q: " << std::endl << this->_Q << std::endl;
    
    // cv::Mat mapXL, mapYL;
    // cv::Mat mapXR, mapYR;
    cv::initUndistortRectifyMap(this->_cameraMatrixL, this->_distCoeffL, R1, P1, this->_imageSize, CV_32FC1, this->_mapXL, this->_mapYL);
    cv::initUndistortRectifyMap(this->_cameraMatrixR, this->_distCoeffR, R2, P2, this->_imageSize, CV_32FC1, this->_mapXR, this->_mapYR);
}

void StereoCalib::stereoRectify(cv::Mat imageL, cv::Mat imageR, cv::Mat& rectifyImageL, cv::Mat& rectifyImageR){

    //! 经过remap之后，左右相机的图像已经共面并且行对齐
    // cv::Mat rectifyImageL, rectifyImageR;
    cv::remap(imageL, rectifyImageL, this->_mapXL, this->_mapYL, cv::INTER_LINEAR);
    cv::remap(imageR, rectifyImageR, this->_mapXR, this->_mapYR, cv::INTER_LINEAR);

    // cv::rectangle(rectifyImageL, this->_validRoi[0], cv::Scalar(255), 2);
    // cv::rectangle(rectifyImageR, this->_validRoi[1], cv::Scalar(255), 2);

    // // 校验
    // cv::Size showSize(this->_imageSize.width * 2, this->_imageSize.height);
    // cv::Mat showRectifyImg = cv::Mat::zeros(showSize, CV_8UC1);
    // rectifyImageL.copyTo(showRectifyImg(cv::Rect(0, 0, showSize.width / 2.0, showSize.height)));
    // rectifyImageR.copyTo(showRectifyImg(cv::Rect(showSize.width / 2.0, 0, showSize.width / 2.0, showSize.height)));
    // for(int i = 0; i < showSize.height; i += 50){
    //     cv::line(showRectifyImg, cv::Point(0, i), cv::Point(showSize.width-1, i), cv::Scalar(255), 1, cv::LINE_AA);
    // }
    // cv::imwrite("./temp/showRectifyImg.jpg", showRectifyImg);
}

void StereoCalib::stereoSGBM(const cv::Mat rectifyImageL, const cv::Mat rectifyImageR, std::vector<cv::Point2f>& landmarkL, std::vector<cv::Point2f>& landmarkR, cv::Mat& filteredDisparityColorMap, cv::Mat& xyz, std::vector<cv::Point3f>& worldPts){

    // // int mode = cv::StereoSGBM::MODE_SGBM_3WAY;
    cv::Mat showLandmarkL = rectifyImageL.clone();
    cv::Mat showLandmarkR = rectifyImageR.clone();
    for(int i = 0; i < landmarkL.size(); i++){
        cv::circle(showLandmarkL, landmarkL[i], 1, cv::Scalar(255), -1);
        cv::circle(showLandmarkR, landmarkR[i], 1, cv::Scalar(255), -1);
    }
    cv::imwrite("./temp/showLandmarkL.jpg", showLandmarkL);
    cv::imwrite("./temp/showLandmarkR.jpg", showLandmarkR);

    

    std::vector<int> index{6, 10, 14, 15, 17};
    //! ### 1
    for(int i = 0; i < index.size(); i++){
        int t = index[i];
        cv::Point3f pt1(landmarkL[t].x, landmarkL[t].y, abs(landmarkL[t].x-landmarkR[t].x));
        cv::Point3f out = this->calculateDistance(pt1);
        if(t == 14){
            std::cout << "dist1: " << out << std::endl;
        }
        worldPts.push_back(out);
    }

    // //! ### 2
    // int baseline = 60; // mm
    // double focalLen = 2080;
    // for(int i = 0; i < index.size(); i++){
    //     int t = index[i];
    //     double distance = (baseline * focalLen) / (abs(landmarkL[t].x-landmarkR[t].x));
    //     std::cout << "dist2: " << distance << std::endl;
    // }

    // //! ### 3
    // cv::Mat point3d;
    // cv::reprojectImageTo3D(disparityL, point3d, this->_Q, true);
    // for(int i = 0; i < index.size(); i++){
    //     int t = index[i];
    //     double x = point3d.at<cv::Vec3f>(landmarkL[t].y, landmarkL[t].x)[0];
    //     double y = point3d.at<cv::Vec3f>(landmarkL[t].y, landmarkL[t].x)[1];
    //     double z = point3d.at<cv::Vec3f>(landmarkL[t].y, landmarkL[t].x)[2];
    //     std::cout << "dist3: " << "x: " << x << ", y: " << y << ", z: " << z << std::endl;
    // }
    //############################### 20231113 #######################################
}

void StereoCalib::projectPoint(const std::vector<cv::Point3f>& worldPts, std::vector<cv::Point2f>& imagePts){
    for(const auto& pt : worldPts){
        cv::Mat_<double> pt3d = (cv::Mat_<double>(3, 1) << pt.x, pt.y, pt.z);
        cv::Mat_<double> imagePoint = this->_cameraMatrixL * cv::Mat(pt3d);
        imagePts.push_back(cv::Point2f(imagePoint.at<double>(0, 0)/imagePoint.at<double>(2, 0), imagePoint.at<double>(1, 0)/imagePoint.at<double>(2, 0)));
    }
}

void StereoCalib::calcStereoDist(double lx, double ly, double rx, double ry){
    double x = 0;
	double y = 0;
	double z = 0;

    cv::Mat c1 = this->_cameraMatrixL;
    cv::Mat c2 = this->_cameraMatrixR;
    cv::Mat r = this->_R;
    cv::Mat t = this->_T;

    double m1[9] = {c1.at<double>(0, 0), c1.at<double>(1, 0), c1.at<double>(2, 0), 
                    c1.at<double>(0, 1), c1.at<double>(1, 1), c1.at<double>(2, 1), 
                    c1.at<double>(0, 2), c1.at<double>(1, 2), c1.at<double>(2, 2)};
    double m2[9] = {c2.at<double>(0, 0), c2.at<double>(1, 0), c2.at<double>(2, 0), 
                    c2.at<double>(0, 1), c2.at<double>(1, 1), c2.at<double>(2, 1), 
                    c2.at<double>(0, 2), c2.at<double>(1, 2), c2.at<double>(2, 2)};
    double R[9] = { r.at<double>(0, 0), r.at<double>(1, 0), r.at<double>(2, 0), 
                    r.at<double>(0, 1), r.at<double>(1, 1), r.at<double>(2, 1), 
                    r.at<double>(0, 2), r.at<double>(1, 2), r.at<double>(2, 2)};
    double T[3] = {t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)};

    double numerator = (m1[0] + m1[4]) / 2 * ((m2[0] + m2[4]) / 2 * T[0] - (rx - m2[6])*T[2]);
	double denominator1 = (rx - m2[6])*(R[6] * (lx - m1[6]) + R[7] * (ly - m1[7]) + R[8] * (m1[0] + m1[4]) / 2);
	double denominator2 = (m2[0] + m2[4]) / 2 * (R[0] * (lx - m1[6]) + R[1] * (ly - m1[7]) + R[2] * (m1[0] + m1[4]) / 2);
	z = numerator / (denominator1 - denominator2);
	x = z * (lx - m1[6]) / ((m1[0] + m1[4]) / 2);
	y = z * (ly - m1[7]) / ((m1[0] + m1[4]) / 2);
	
	printf("该点世界坐标为 %f\t %f\t %f\n", x, y, z);
}

cv::Point3f StereoCalib::calculateDistance(cv::Point3f in){
    std::vector<cv::Point3f> input;
    std::vector<cv::Point3f> output;

    input.push_back(in);

    /* Transform 2D coordinates to 3D coordinates, using Q matrix */
    cv::perspectiveTransform(input, output, this->_Q);

    return output[0];
}