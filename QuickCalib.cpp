#ifdef __linux__
#include <unistd.h>
#elif _WIN32
#include <windows.h>
#endif

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "Player/Controller.h"
#include "Player/ChannelObserver.h"
#include "Player/Player.h"
#include "Collector/Collector.h"
#include "QCamCalib/QCamCalib.h"
#include "StereoCalib/StereoCalib.h"

#include "CalibSys/CalibSys.h"

#define QUICKCALIB_SHORTCUT_QUIT    0x71    // 'q'
#define QUICKCALIB_SHORTCUT_START   0x73    // 's'

void test_CalibSys(){

    // std::vector<int> cameras{0, 1};
    // CalibSys calibSys(CalibSys::InputType::TYPE_CAMERA, cameras, cv::Size(1920, 1080));

    std::vector<std::string> videos{
        "./example/calib_8/videoL.mp4",
        "./example/calib_8/videoR.mp4"
    };
    CalibSys calibSys(CalibSys::InputType::TYPE_VIDEO, videos, cv::Size(1920, 1080));

    calibSys.init();
    calibSys.run();
}

void test_QuickCalib(){


    std::cout << "====== QuickCalib ======" << std::endl;

    std::string videoL{"./example/calib_imgs_5_video/outputL1.mp4"};
    std::string videoR{"./example/calib_imgs_5_video/outputR1.mp4"};

    std::vector<std::string> videos{videoL, videoR};
    cv::Size frameSize(1920, 1080);

    QCamCalib::Config camCalibConfig;
    camCalibConfig.camType = QCamCalib::CamType::CAM_GENERAL;
    camCalibConfig.patternSize = cv::Size(8, 5);
    camCalibConfig.squareSize = cv::Size(30, 30);
    camCalibConfig.srcImgSize = cv::Size(1920, 1080);
    QCamCalib camCalib;
    if(!camCalib.init(camCalibConfig)){
        printf("QCamCalib init fail!\n");
    }
    printf("QCamCalib init successfully!\n");

    StereoCalib stereoCalib(frameSize);

    Controller controller(frameSize);
    Collector collector(videos.size());

    for(int i = 0; i < videos.size(); i++){
        ChannelObserver* channel = new Player(videos[i]);
        controller.attach(channel);
    }

    bool startCalibFlag = false;
    bool startStereoCalibFlag = false;
    int loop = -1;
    while(true){
        loop++;

        controller.notify(frameSize);

        std::vector<cv::Mat>& currFrames = controller.getCurrFrame();
        // cv::Mat frameL, frameR;
        // frameL = currFrames[0];
        // frameR = currFrames[1];
        // cv::imwrite("./temp/frameL.jpg", frameL);
        // cv::imwrite("./temp/frameR.jpg", frameR);

        if(startCalibFlag){
            controller.setTitle("Start Calibrate!", 2);
            collector.startCollect();
            startCalibFlag = false;
        }

        if(startStereoCalibFlag){
            
            // mono calib
            std::vector<cv::Mat> cameraMatrixs(videos.size());
            std::vector<cv::Mat> distCoeffs(videos.size());
            std::vector<std::vector<std::vector<cv::Point2f>>>& corners = collector.getCorners();
            for(int i = 0; i < corners.size(); i++){
                camCalib.calibrateCamera(corners[i], cameraMatrixs[i], distCoeffs[i]);
            }

            stereoCalib.setIntrisicsL(cameraMatrixs[0], distCoeffs[0]);
            stereoCalib.setIntrisicsR(cameraMatrixs[1], distCoeffs[1]);

            stereoCalib.stereoCalibrate(corners[0], corners[1]);

            startStereoCalibFlag = false;
        }


        int ret = collector.inputImage(currFrames);
        if(ret == 1){ // 阶段1
            controller.setBackgroundColor(cv::Scalar(0, 255, 0), 30);
        }else if(ret == 2){ // yellow
            controller.setBackgroundColor(cv::Scalar(0, 255, 255), 30);
        }else if(ret == 3){
            controller.setBackgroundColor(cv::Scalar(0, 0, 255), 30);
        }else if(ret == 4){
            controller.setBackgroundColor(cv::Scalar(0, 0, 0), 1);
        }else if(ret == 5){
            // collect finish.
            printf("collect finish.\n");
            startStereoCalibFlag = true;
            controller.setTitle("Begin Stereo Calibrate!", 3);
            controller.notify(frameSize);
        }


#ifdef __linux__

        if(loop == 20){
            printf("loop: %d\n", loop);
            startCalibFlag = true;
        }

        cv::imwrite("./temp/showMat.jpg", controller.getShowMat());
        usleep(200 * 1000); 
#elif _WIN32
        cv::imshow("showMat.jpg", controller.getShowMat());
        int key = cv::waitKey(20);
        if(key == QUICKCALIB_SHORTCUT_QUIT){
            printf("Quit!\n");
            break;
        }else if(key == QUICKCALIB_SHORTCUT_START){
            startCalibFlag = true;
        }
#endif



    }

    controller.release();

    return;

//     cv::VideoCapture capL(videoL);
//     if(!capL.isOpened()){
//         printf("open video fail!(%s)\n", videoL.c_str());
//         return;
//     }

//     cv::VideoCapture capR(videoR);
//     if(!capR.isOpened()){
//         printf("open video fail!(%s)\n", videoR.c_str());
//         return;
//     }

//     int frameW = capL.get(cv::CAP_PROP_FRAME_WIDTH);
//     int frameH = capL.get(cv::CAP_PROP_FRAME_HEIGHT);
//     int fps = int(capL.get(cv::CAP_PROP_FPS));
//     printf("frameW: %d\n", frameW);
//     printf("frameH: %d\n", frameH);
//     printf("fps: %d\n", fps);

//     double scale = 0.5;

//     cv::Mat frameL;
//     cv::Mat frameR;
//     cv::Mat showMat;
//     bool startFlag = false;
//     while(true){    
//         capL >> frameL;
//         capR >> frameR;

//         if(frameL.empty() || frameR.empty()) break;
//         cv::resize(frameL, frameL, cv::Size(), scale, scale);
//         cv::resize(frameR, frameR, cv::Size(), scale, scale);
//         showMat = cv::Mat(frameL.rows, frameL.cols * 2, frameL.type());

//         frameL.copyTo(showMat(cv::Rect(0, 0, frameL.cols, frameL.rows)));
//         frameR.copyTo(showMat(cv::Rect(frameL.cols, 0, frameR.cols, frameR.rows)));

//         if(startFlag){
//             std::string text = "Start Calibrate!";
//             int fontFace = cv::FONT_HERSHEY_SIMPLEX;
//             double fontScale = 2.5;
//             int thickness = 2;
//             cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, nullptr);
//             cv::Point textOrigin((showMat.cols - textSize.width) / 2.0, (showMat.rows + textSize.height) / 2.0);
//             cv::putText(showMat, text, textOrigin, fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
//         }

// #ifdef _WIN32
//         cv::imshow("showMat", showMat);
//         int key = cv::waitKey(50);
//         if(key == QUICKCALIB_SHORTCUT_QUIT){
//             printf("Quit!\n");
//             break;
//         }else if(key == QUICKCALIB_SHORTCUT_START){
//             startFlag = true;
//         }
// #endif

//     }

//     capL.release();
//     capR.release();
}