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
    Demo demo(Demo::CalibSrcType::TYPE_PHOTO);

    demo.run();
}

void test_monoCalib();
void test_stereoCalib();

int main(int, char**){
    std::cout << "Hello, from StereoCalib!\n";

    // test_StereoCalib();

    runDemo();

    // test_monoCalib();

    // test_stereoCalib();

    return 0;
}
