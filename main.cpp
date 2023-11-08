#include <iostream>

#include "StereoCalib/StereoCalib.h"

void test_StereoCalib(){
    std::string imgLPath{"./example/imgL.jpg"};
    std::string imgRPath{"./example/imgR.jpg"};

    StereoCalib stereoCalib;

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

    cv::Mat rectifyImageL;
    cv::Mat rectifyImageR;
    stereoCalib.stereoRectify(grayImgL, grayImgR, rectifyImageL, rectifyImageR);
    cv::imwrite("./temp/rectifyImageL.jpg", rectifyImageL);
    cv::imwrite("./temp/rectifyImageR.jpg", rectifyImageR);

    cv::Mat disparity;
    stereoCalib.stereoSGBM(rectifyImageL, rectifyImageR, disparity);
}

int main(int, char**){
    std::cout << "Hello, from StereoCalib!\n";

    test_StereoCalib();

    return 0;
}
