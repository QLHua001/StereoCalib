#include <iostream>
#include <dirent.h>

#include "StereoCalib/StereoCalib.h"

bool searchSpecifiedFiles(std::string folder, std::string extension, std::vector<std::string>& filePaths){

    if(folder.empty() || extension.empty()) return false; 

    if(folder.back() != '/'){
        folder += "/";
    }

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
