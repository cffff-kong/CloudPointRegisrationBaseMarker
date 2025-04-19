cmake_minimum_required(VERSION 3.12)
project(CloudPointRegistration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置正确的 OpenCV_DIR（一定要指向 lib 文件夹，不是根目录）
set(OpenCV_DIR "D:/opencv4.7/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# PCL
set(PCL_DIR "D:/PCL 1.14.1/cmake")
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/CloudPointProcess.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARY_DIRS})

