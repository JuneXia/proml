cmake_minimum_required(VERSION 3.6.3)
project(cppml)

set(CMAKE_CXX_STANDARD 11)

set(Torch_DIR /home/xiajun/program/libtorch/share/cmake/Torch)   # 根据自己保存的路径输入
#set(OpenCV_DIR /home/xiajun/program/opencv-3.4.0/build)   # 我的opencv之前已经安装过，并且已安装到/usr/local/lib系统路径下

find_package(Torch REQUIRED)   # 查找库
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

add_executable(cppml main.cpp maskface.cpp maskface.h example.cpp)

target_link_libraries(cppml ${TORCH_LIBRARIES} ${OpenCV_LIBS})   # 添加链接文件