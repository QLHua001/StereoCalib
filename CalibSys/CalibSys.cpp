
#include <opencv2/opencv.hpp>
#include "CalibSys/CalibSys.h"

#include "Player/ChannelObserver.h"
#include "Player/Player.h"

#ifdef __linux__
#include <unistd.h>
#endif 

#define QUICKCALIB_SHORTCUT_QUIT    0x71    // 'q'
#define QUICKCALIB_SHORTCUT_START   0x73    // 's'

CalibSys::CalibSys(InputType inputType, std::vector<int>& cameras, cv::Size imageSize){
    this->_inputType = inputType;
    this->_cameras.insert(this->_cameras.end(), cameras.begin(), cameras.end());
    this->_imageSize = imageSize;
}

CalibSys::CalibSys(InputType inputType, std::vector<std::string>& videos, cv::Size imageSize){
    this->_inputType = inputType;
    this->_videos.insert(this->_videos.end(), videos.begin(), videos.end());
    this->_imageSize = imageSize;
}

CalibSys::~CalibSys(){
    this->_controller.release();
}

void CalibSys::init(){

    cv::Size frameSize = this->_imageSize;

    this->_controller = Controller(frameSize);
    this->_controller.setImgSize(this->_imageSize);
    switch(this->_inputType){
    case InputType::TYPE_VIDEO:{

        for(int i = 0; i < this->_videos.size(); i++){
            ChannelObserver* channel = new Player(this->_videos[i]);
            this->_controller.attach(channel);
        }
        this->_count = this->_videos.size();
        break;
    }
    case InputType::TYPE_CAMERA:{

        for(int i = 0; i < this->_cameras.size(); i++){
            ChannelObserver* channel = new Player(this->_cameras[i]);
            this->_controller.attach(channel);
        }
        this->_count = this->_cameras.size();
        break;
    }
    }

    // std::string videoL{"./example/calib_imgs_5_video/outputL1.mp4"};
    // std::string videoR{"./example/calib_imgs_5_video/outputR1.mp4"};

    // std::vector<std::string> videos{videoL, videoR};

    

    double scale = 0.5;
    cv::Size patternSize(8, 5);
    cv::Size squareSize(30, 30);
    QCamCalib::Config camCalibConfig;
    camCalibConfig.camType = QCamCalib::CamType::CAM_GENERAL;
    camCalibConfig.patternSize = patternSize;
    camCalibConfig.squareSize = squareSize;
    camCalibConfig.srcImgSize = frameSize;
    camCalibConfig.scale = scale;
    if(!this->_camCalib.init(camCalibConfig)){
        printf("QCamCalib init fail!\n");
    }
    printf("QCamCalib init successfully!\n");

    this->_stereoCalib = StereoCalib(frameSize, scale, patternSize, squareSize);

    this->_collector = Collector(this->_count);
    this->_collector.setCamCalibConfig(camCalibConfig);

}

bool CalibSys::run(){

    bool startCalibFlag = false;
    bool startStereoCalibFlag = false;
    int loop = -1;
    int index = 0;
    while(true){
        loop++;

        this->_controller.notify(this->_imageSize);

        std::vector<cv::Mat>& currFrames = this->_controller.getCurrFrame();
        // cv::Mat frameL, frameR;
        // frameL = currFrames[0];
        // frameR = currFrames[1];
        // cv::imwrite("./temp/frameL.jpg", frameL);
        // cv::imwrite("./temp/frameR.jpg", frameR);

        if(startCalibFlag){
            this->_controller.setTitle("Start Calibrate!", 2);
            this->_collector.startCollect();
            index += 1;
            startCalibFlag = false;
        }

        if(startStereoCalibFlag){
            
            // mono calib
            std::vector<cv::Mat> cameraMatrixs(this->_count);
            std::vector<cv::Mat> distCoeffs(this->_count);
            std::vector<std::vector<std::vector<cv::Point2f>>>& corners = this->_collector.getCorners();
            for(int i = 0; i < corners.size(); i++){
                this->_camCalib.calibrateCamera(corners[i], cameraMatrixs[i], distCoeffs[i]);
            }

            this->_stereoCalib.setIntrisicsL(cameraMatrixs[0], distCoeffs[0]);
            this->_stereoCalib.setIntrisicsR(cameraMatrixs[1], distCoeffs[1]);

            this->_stereoCalib.stereoCalibrate(corners[0], corners[1]);

            char filename[128];
            sprintf(filename, "./config/Params.yml");
            this->_stereoCalib.saveParams(filename);

            startStereoCalibFlag = false;

            return true;
        }


        int ret = this->_collector.inputImage(currFrames);
        if(ret == 1){ // 阶段1
            this->_controller.setBackgroundColor(cv::Scalar(0, 255, 0), 30);
        }else if(ret == 2){ // yellow
            this->_controller.setBackgroundColor(cv::Scalar(0, 255, 255), 30);
        }else if(ret == 3){
            this->_controller.setBackgroundColor(cv::Scalar(0, 0, 255), 30);
        }else if(ret == 4){
            this->_controller.setBackgroundColor(cv::Scalar(0, 0, 0), 1);
        }else if(ret == 5){
            // collect finish.
            printf("collect finish.\n");
            startStereoCalibFlag = true;
            this->_controller.setTitle("Begin Stereo Calibrate!", 3);
            this->_controller.notify(this->_imageSize);
        }


#ifdef __linux__

        if(loop == 20){
            printf("loop: %d\n", loop);
            startCalibFlag = true;
        }

        cv::imwrite("./temp/showMat.jpg", this->_controller.getShowMat());
        usleep(20 * 1000); 
#elif _WIN32
        cv::imshow("showMat.jpg", this->_controller.getShowMat());
        int key = cv::waitKey(20);
        if(key == QUICKCALIB_SHORTCUT_QUIT){
            printf("Quit!\n");
            break;
        }else if(key == QUICKCALIB_SHORTCUT_START){
            startCalibFlag = true;
        }
#endif



    }

}
