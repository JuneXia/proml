cmake_minimum_required(VERSION 3.0)  # 2.8
project(cppml_sdk_test)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(Torch_DIR ${PROJECT_SOURCE_DIR}/ml/libtorch/share/cmake/Torch)
set(OpenCV_DIR /home/xiajun/program/opencv-3.4.1/install/share/OpenCV)

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

include_directories(./inc)

message(STATUS "    link path: ${PROJECT_SOURCE_DIR}/linux_a")
link_directories(${PROJECT_SOURCE_DIR}/linux_a)

add_executable(cppml_demo test.cpp)

target_link_libraries(cppml_demo ${TORCH_LIBRARIES} ${OpenCV_LIBS} libgcai_maskface.a)
