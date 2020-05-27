//
// Created by xiajun on 20-3-17.
//

#ifndef CPPML_MASKFACE_H
#define CPPML_MASKFACE_H

#include <iostream>
#include <torch/script.h>
//#include <torch/torch.h>  // dispensible on ubuntu, but must not on windows.
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>

class maskface {
private:
    torch::jit::script::Module _module;
    int _margin;
    int _imshape[3];

public:
    /*构造函数：你需要传入模型所在路径*/
    maskface(const std::string& filename);

    /*function: 预测函数
     *  params:
     *      images: rgb image.
     *      bboxes: [b, 4], b代表一张图片可以有多个人脸框，每个人脸框包含四个数值，分表表示 (x_left, y_left, width, height)
     *  return:
     *      if param_error:
     *          throw
     *      elif runtime_error:
     *          return empty vector.
     *      else:
     *          predict result, [ {label: confidence}, ... ]*/
    std::vector<std::map<int, float>> predict(cv::Mat images, std::vector<std::vector<int>> bboxes);

    ~maskface();
};


#endif //CPPML_MASKFACE_H
