cmake_minimum_required(VERSION 2.8)
set(PROJECT_NAME opencl_app)
project(${PROJECT_NAME})

find_package(OpenCL REQUIRED)

set(WITH_OPENCL ON)
set(WITH_CUDA OFF)
set(WITH_CUFFT OFF)
set(BUILD_TESTS OFF)
set(BUILD_opencv_python2 OFF)
set(BUILD_opencv_python3 OFF)
add_subdirectory(opencv)

set(OpenCV_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/opencv/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/core/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/highgui/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/imgcodecs/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/imgproc/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/videoio/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/dnn/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/calib3d/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/features2d/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/flann/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/ml/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/objdetect/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/photo/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/shape/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/stitching/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/superres/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/video/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/videostab/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/viz/include"
    "${CMAKE_SOURCE_DIR}/opencv/modules/ocl/include"
    "${CMAKE_BINARY_DIR}")

include_directories(${OpenCV_INCLUDE_DIRS} ${OpenCL_INCLUDE_DIRS})
add_executable(opencl_app oclmat_sample.cpp)
target_link_libraries(opencl_app opencv_core opencv_highgui ${OpenCL_LIBRARIES})

add_executable(demo demo.cpp)
target_link_libraries(demo opencv_core opencv_highgui opencv_dnn ${OpenCL_LIBRARIES})

add_executable(demo2 demo2.cpp)
target_link_libraries(demo2 opencv_superres opencv_objdetect opencv_core opencv_highgui ${OpenCL_LIBRARIES})
