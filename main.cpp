#include <iostream>

#ifdef __linux__
#include <dirent.h>
#elif _WIN32
#include <io.h>
#endif

#include "StereoCalib/StereoCalib.h"
#include "QNN/Predictor/DataTypeDMS.h"
#include "QNN/AIDetector/AIDetector.h"

#include "Demo/Demo.h"

bool searchSpecifiedFiles(std::string folder, std::string extension, std::vector<std::string>& filePaths){

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

void getCalibImgs(const std::string& imgRoot, std::vector<std::string>& imgLPathList, std::vector<std::string>& imgRPathList){

    std::vector<std::string> filePaths;
    searchSpecifiedFiles(imgRoot, ".jpg", filePaths);
    printf("Found %ld imgs\n", filePaths.size());

    int imgNum = filePaths.size();
    for(int i = 0; i < imgNum/2; i++){
        char basename[128];

        sprintf(basename, "left%d.jpg", i);
        imgLPathList.push_back(imgRoot + basename);

        sprintf(basename, "right%d.jpg", i);
        imgRPathList.push_back(imgRoot + basename);
    }

}

void test_StereoCalib(){
    std::string imgRoot{"./example/calib_imgs/"};

    std::string imgLPath{"./example/calib_imgs/left1.jpg"};
    std::string imgRPath{"./example/calib_imgs/right1.jpg"};

    AIDetector* AiDMSMTYolox = new AIDMSMTYolox;
    AiDMSMTYolox->init();

    AIDetector* AiDMSMTFace = new AIDMSMTFace;
    AiDMSMTFace->init();

    std::vector<std::string> imgLPathList, imgRPathList;
    getCalibImgs(imgRoot, imgLPathList, imgRPathList);

    StereoCalib stereoCalib;

    stereoCalib.stereoCalibrate(imgLPathList, imgRPathList);

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

    cv::Mat gray888;
    cv::cvtColor(grayImgL, gray888, cv::COLOR_GRAY2BGR);
    cv::imwrite("./temp/gray888.jpg", gray888);

    Input DMSMTYoloxInput;
    DMSMTYoloxInput.data = gray888.data;
    DMSMTYoloxInput.width = gray888.cols;
    DMSMTYoloxInput.height = gray888.rows;
    DMSMTYoloxInput.format = ImgFormat::FMT_BGR888;
    DMSMTYolox DMSMTYoloxOutput;
    AiDMSMTYolox->run(&DMSMTYoloxInput, &DMSMTYoloxOutput);

    cv::Mat rectifyImageL;
    cv::Mat rectifyImageR;
    stereoCalib.stereoRectify(grayImgL, grayImgR, rectifyImageL, rectifyImageR);
    cv::imwrite("./temp/rectifyImageL.jpg", rectifyImageL);
    cv::imwrite("./temp/rectifyImageR.jpg", rectifyImageR);

    // cv::Mat filteredDisparityColorMap;
    // cv::Mat xyz;
    // stereoCalib.stereoSGBM(rectifyImageL, rectifyImageR, filteredDisparityColorMap, xyz);

    delete AiDMSMTFace;
    delete AiDMSMTYolox;
}

void runDemo(){

    std::vector<int> cameraId{0, 1}; // camera id
    Demo demo(cameraId);

    // std::string videoL{"./example/calib_imgs_5_video/outputL1.mp4"};
    // std::string videoR{"./example/calib_imgs_5_video/outputR1.mp4"};
    std::string outputPath{""};

    cv::VideoCapture capL(0);
    if(!capL.isOpened()){
        printf("open video fail!\n");
        return;
    }

    cv::VideoCapture capR(1);
    if(!capR.isOpened()){
        printf("open video fail!\n");
        return;
    }

    int frameW = capL.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameH = capL.get(cv::CAP_PROP_FRAME_HEIGHT);
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
        if(imgL.empty() || imgR.empty()) break;

        // int tarW = 1920 * 0.5;
        // int tarH = 1080 * 0.5;

        // if((imgL.cols != tarW) || (imgL.rows != tarH)){
        //     cv::resize(imgL, imgL, cv::Size(tarW, tarH));
        // }
        // if((imgR.cols != tarW) || (imgR.rows != tarH)){
        //     cv::resize(imgR, imgR, cv::Size(tarW, tarH));
        // }

        // imgL = testImgL;
        // imgR = testImgR;
        if(imgL.empty() || imgR.empty()) break;

        demo.run(imgL, imgR);
    }
    if(!outputPath.empty()){
        writer.release();
    }
    capL.release();
    capR.release();

}

void test_video(){
    std::string outputDir{"./example/"};
    std::string outputL = outputDir + "videoL.mp4";
    std::string outputR = outputDir + "videoR.mp4";

    cv::VideoWriter writerL;
    cv::VideoWriter writerR;

    cv::VideoCapture capL(0);
    if(!capL.isOpened()){
        printf("cv::VideoCapture open fail!(%d)\n", 0);
        return;
    }

    cv::VideoCapture capR(1);
    if(!capR.isOpened()){
        printf("cv::VideoCapture open fail!(%d)\n", 1);
        return;
    }

    int frameWL = capL.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHL = capL.get(cv::CAP_PROP_FRAME_HEIGHT);
    // int fpsL = int(capL.get(cv::CAP_PROP_FPS));
    int fpsL = 15;

    if(!outputL.empty()){
        writerL.open(outputL, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsL, cv::Size(frameWL, frameHL));
        if(!writerL.isOpened()){
            printf("writerL open fail!\n");
            return;
        }
    }

    int frameWR = capR.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHR = capR.get(cv::CAP_PROP_FRAME_HEIGHT);
    // int fpsR = int(capR.get(cv::CAP_PROP_FPS));
    int fpsR = 15;

    if(!outputR.empty()){
        writerR.open(outputR, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsR, cv::Size(frameWR, frameHR));
        if(!writerR.isOpened()){
            printf("writerR open fail!\n");
            return;
        }
    }

    cv::Mat frameL, frameR;
    while(true){
        
        capL >> frameL;
        capR >> frameR;
        if(frameL.empty() || frameR.empty()) break;

        if(!outputL.empty()) writerL << frameL;
        if(!outputR.empty()) writerR << frameR;

        cv::imshow("frameL", frameL);
        cv::imshow("frameR", frameR);
        int key = cv::waitKey(30);
        if(key == 'q') break;
    }

    if(!outputL.empty()) writerL.release();
    if(!outputR.empty()) writerR.release();

    capL.release();
    capR.release();
}

void test_monoCalib();
void test_stereoCalib();
void test_QuickCalib();
void test_CalibSys();

int main(int, char**){
    std::cout << "Hello, from StereoCalib!\n";

    // test_StereoCalib();

    runDemo();

    // test_monoCalib();

    // test_stereoCalib();

    // test_QuickCalib();

    // test_CalibSys();

    // test_video();

    return 0;
}
