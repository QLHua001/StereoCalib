cmake_minimum_required(VERSION 3.0.0)
project(StereoCalib VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

set(CMAKE_BUILD_TYPE Debug)

set(EXECUTABLE_OUTPUT_PATH ../output/)

if(CMAKE_HOST_WIN32)

elseif(CMAKE_HOST_UNIX)

    # OpenMP
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        message(FATAL_ERROR "OpenMP Not Found!")
    endif ()

    set(OpenCV_DIR "/QMy-File2/QPackage/opencv-4.8.0/output_host_ubuntu2004_contrib/")
    include_directories(${OpenCV_DIR}/include/opencv4/)
    link_directories(${OpenCV_DIR}/lib/)
    set(OpenCV_LIBS libopencv_world.so.4.8.0)

    set(NCNN_DIR "/QMy-File2/QPackage/ncnn-20220322/install/")
    include_directories(${NCNN_DIR}/include/)
    link_directories(${NCNN_DIR}/lib/)
    set(NCNN_LIBS libncnn.a)

endif()

include_directories(./)
set(SRC_Demo
    ./Demo/Demo.cpp)

set(SRC_StereoCalib
    ./StereoCalib/StereoCalib.cpp)

set(SRC_QCamCalib
    QCamCalib/QCamCalib.cpp)

set(SRC_QNN
    QNN/Predictor/Predictor.cpp
    QNN/Predictor/PreProcessor.cpp
    QNN/Predictor/NetNCNN.cpp
    QNN/Predictor/PostDMSMTFace.cpp
    QNN/Predictor/PostDMSMTYolox.cpp
    QNN/AIDetector/AIDMSMTFace.cpp
    QNN/AIDetector/AIDMSMTYolox.cpp)

set(SRC_Player 
    Player/Player.cpp
    Player/Controller.cpp)

set(SRC_Collector
    Collector/Collector.cpp)

set(SRC_CalibSys
    CalibSys/CalibSys.cpp)

set(SRC_Utils
    monoCalib.cpp
    stereoCalib.cpp
    QuickCalib.cpp)


set(LIB_LIST ${OpenCV_LIBS} ${NCNN_LIBS})
set(SRC_LIST main.cpp ${SRC_Demo} ${SRC_StereoCalib} ${SRC_QCamCalib} ${SRC_QNN} ${SRC_Player} ${SRC_Collector} ${SRC_CalibSys} ${SRC_Utils})

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${LIB_LIST})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
