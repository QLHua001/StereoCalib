#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static const int boardWidth = 8;                               //横向的角点数目  
static const int boardHeight = 5;                              //纵向的角点数据  
static const int boardCorner = boardWidth * boardHeight;       //总的角点数据  
static const int frameNumber = 24;                             //相机标定时需要采用的图像帧数  
static const int squareSize = 30;                              //标定板黑白格子的大小 单位mm  
static const Size boardSize = Size(boardWidth, boardHeight);   //总的内角点
 
static Mat intrinsic;                                                //相机内参数  
static Mat distortion_coeff;                                   //相机畸变参数  
static vector<Mat> rvecs;                                        //旋转向量  
static vector<Mat> tvecs;                                        //平移向量  
static vector<vector<Point2f>> corners;                        //各个图像找到的角点的集合 和objRealPoint 一一对应  
static vector<vector<Point3f>> objRealPoint;                   //各副图像的角点的实际物理坐标集合  
static vector<Point2f> corner;                                   //某一副图像找到的角点  

/*计算标定板上模块的实际物理坐标*/
static void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
 vector<Point3f> imgpoint;
 for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
 {
 for (int colIndex = 0; colIndex < boardwidth; colIndex++)
 {
 imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
 }
 }
 for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
 {
 obj.push_back(imgpoint);
 }
}

void test_monoCalib(){
    std::cout << "=== test_monoCalib() ===" << std::endl;

    int imageHeight = 1080;     //图像高度
    int imageWidth = 1920;      //图像宽度
    int goodFrameCount = 0;    //有效图像的数目
    std::string prefix = "left";
    std::string testImgPath = "./example/calib_imgs_test/left_4.jpg";

    std::string outputYmlPath = "./config/intrinsicL.yml";

    Mat rgbImage,grayImage;

    while (goodFrameCount < frameNumber){
        char filename[100];
        sprintf(filename, "./example/calib_imgs_4/%s_%d.jpg", prefix.c_str(), goodFrameCount);

        rgbImage = imread(filename);
        if (rgbImage.empty())
        {
            printf("Could not load rgbImage...\n");
            return;
        }

        cvtColor(rgbImage, grayImage, cv::COLOR_BGR2GRAY);

        bool isFind = findChessboardCorners(rgbImage, boardSize, corner, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
        if (isFind == true){ //所有角点都被找到 说明这幅图像是可行的  
            //精确角点位置，亚像素角点检测
            cornerSubPix(grayImage, corner, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 20, 0.1));
            //绘制角点
            drawChessboardCorners(rgbImage, boardSize, corner, isFind);
            char outputName[512];
            sprintf(outputName, "./temp/chessboard_%d.jpg", goodFrameCount);
            imwrite(outputName, rgbImage);
            corners.push_back(corner);
            /*cout << "The image" << goodFrameCount << " is good" << endl;*/
            printf("The image %d is good...\n", goodFrameCount);
            goodFrameCount++;
        }else{
            printf("The image is bad please try again...\n");
        }
    }

    /*计算实际的校正点的三维坐标*/
    calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
    printf("calculate real successful...\n");

    /*标定摄像头*/
    calibrateCamera(objRealPoint, corners, Size(imageWidth, imageHeight), intrinsic, distortion_coeff, rvecs, tvecs, 0);
    printf("calibration successful...\n");
    std::cout << "intrinsic: " << std::endl << intrinsic << std::endl;
    std::cout << "distortion_coeff: " << std::endl << distortion_coeff << std::endl;

    /*显示畸变校正效果*/
    Mat testImg = imread(testImgPath);
    Mat cImage; 
    undistort(testImg, cImage, intrinsic, distortion_coeff);  //矫正相机镜头变形
    cv::imwrite("./testImg_undistort.jpg", cImage);

    FileStorage file(outputYmlPath, FileStorage::WRITE);
    file << "cameraMatrix" << intrinsic;
    file << "distCoeffs" << distortion_coeff;
    file.release();
}