cmake_minimum_required(VERSION 3.0)  # 2.8
project(cppml_sdk_demo)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")  # 在编译时告诉编译器产生与位置无关的代码(Position-Independent Code)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")

set(Torch_DIR /home/xiajun/program/libtorch/share/cmake/Torch)
# set(OpenCV_DIR /home/xiajun/program/opencv-3.4.1-install/share/OpenCV)  # 用自己指定的编译目录也是ok的，或者直接使用 find_package 搜索OpenCV目录也可以
set(OpenCV_DIR /home/xiajun/program/opencv-4.2.0/static_install/lib/cmake/opencv4)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV Not Found!")
endif(NOT OpenCV_FOUND)

message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(./inc
                    ../inc
                    ${TORCH_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                   )

aux_source_directory(. DIR_SRCS)

include_directories(./inc)

link_directories(${PROJECT_SOURCE_DIR}/linux_a)

add_executable(cppml_demo test.cpp)

target_link_libraries(cppml_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS} libgcai_maskface.a)


