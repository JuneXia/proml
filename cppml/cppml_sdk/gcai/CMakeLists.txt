cmake_minimum_required(VERSION 2.8)  # 指定cmake最小版本
project(cppml_compile_library)       # 给工程取个名字

set(CMAKE_CXX_STANDARD 11)           # 使用c++11标准

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")  # 在编译时告诉编译器产生与位置无关的代码(Position-Independent Code)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

SET(CMAKE_BUILD_TYPE "Debug")
# SET(CMAKE_BUILD_TYPE "Release")

set(Torch_DIR /home/xiajun/program/libtorch/share/cmake/Torch)  # 根据您自己的libtorch放置目录设置
set(OpenCV_DIR /home/xiajun/program/opencv-4.2.0/static_install/lib/cmake/opencv4)  # 根据您自己编译的opencv目录设置


find_package(Torch REQUIRED)   # 查找库
find_package(OpenCV REQUIRED)

if(NOT Torch_FOUND)            # 如果找不到 Torch 则打印出错信息
    message(FATAL_ERROR "Pytorch Not Found!")
endif(NOT Torch_FOUND)

if(NOT OpenCV_FOUND)           # 如果找不到 OpenCV 则打印出错信息
    message(FATAL_ERROR "OpenCV Not Found!")
endif(NOT OpenCV_FOUND)

message(STATUS "Pytorch status:")  # 打印 Torch 库的相关信息
message(STATUS "    libraries: ${TORCH_LIBRARIES}")

message(STATUS "OpenCV library status:")  # 打印 OpenCV 库的相关信息
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 设置编译输出目录
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../linux_a)

# 添加项目当前所在的目录
aux_source_directory(. DIR_LIB_SRCS)

# 指定包含那些头文件目录
include_directories(./inc
                    ../inc
                    ${TORCH_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
                   )


# 链接目录
link_directories(${TORCH_LIBRARIES_DIRS} ${Opencv_LIBRARIES_DIRS})

# 指定使用静态编译，并且指明编译哪些源代码
add_library(gcai_maskface STATIC gc_face_sdk.cpp maskface.cpp)

# 编译完成后，需要做最后的链接，即将我们自己的代码和Torch、OpenCV这些库链接起来
target_link_libraries(gcai_maskface ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# SET_TARGET_PROPERTIES(gcailib PROPERTIES LINKER_LANGUAGE C)
