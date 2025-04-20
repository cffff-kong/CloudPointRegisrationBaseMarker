cmake_minimum_required(VERSION 3.12)
project(CloudPointRegistration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Opencv
#set(OpenCV_DIR "D:/opencv4.7/opencv/build/x64/vc16/lib") #OpenCVConfig路径
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# PCL
#set(PCL_DIR "C:/Program Files/PCL 1.14.0/cmake")
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
# Eigen
include_directories("D:/eigen-3.2.10")

include_directories(include)
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/CloudPointProcess.cpp
    src/SSLReconstruction.cpp
)

target_link_libraries( 
	${PROJECT_NAME} 
    ${OpenCV_LIBS}
	${PCL_LIBRARIES} 
	${CMAKE_THREAD_LIBS_INIT}
)

