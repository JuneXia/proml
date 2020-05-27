//
// Created by xiajun on 20-3-17.
//

#include "maskface.h"


maskface::maskface(const std::string& filename)
{
    _margin = 0;
    _imshape[0] = 96;
    _imshape[1] = 96;
    _imshape[2] = 3;
    _module = torch::jit::load(filename);

    torch::Tensor inputs = torch::zeros({1, _imshape[2], _imshape[0], _imshape[1]});
    inputs = inputs.toType(torch::kFloat);
    _module.forward({inputs}).toTensor();
}


std::vector<std::map<int, float>> maskface::predict(cv::Mat images, std::vector<std::vector<int>> rects)
{
    if(images.empty())
    {
        std::cout << "ERROR: images empty, please check your image!" << std::endl;
        throw images;
    }
    if(rects.size() == 0 or rects[0].size() != 4)
    {
        std::cout << "ERROR: rects empty, please check your rects!" << std::endl;
        throw rects;
    }

    std::vector<std::map<int, float>> ret;
    try
    {
        std::vector<std::vector<int>>::iterator iter1;
        std::vector<int> vec;
        std::vector<cv::Mat> rois;

        // cv::cvtColor(images, images, cv::COLOR_BGR2RGB);

        auto batch_size = rects.size();

        for(iter1 = rects.begin(); iter1 != rects.end(); iter1++)
        {
            cv::Mat roi;
            vec = *iter1;

            vec[0] = std::max(vec[0] - _margin/2, 0);
            vec[1] = std::max(vec[1] - _margin/2, 0);
            vec[2] = std::max(vec[2] + _margin, 0);
            vec[3] = std::max(vec[3] + _margin, 0);

            cv::Rect rect(vec[0], vec[1], vec[2], vec[3]);

            cv::resize(images(rect), roi, cv::Size(_imshape[0], _imshape[1]), 0, 0, cv::INTER_CUBIC);

            rois.push_back(roi);
        }

        torch::Tensor inputs = torch::zeros({long(batch_size), _imshape[2], _imshape[0], _imshape[1]});
        inputs = inputs.toType(torch::kFloat);
        for(int i = 0; i < rois.size(); i++)
        {
//        cv::imshow("show", rois[i]);
//        cv::waitKey();

            torch::Tensor tensor_image = torch::from_blob(rois[i].data, {1, rois[i].rows, rois[i].cols, _imshape[2]}, torch::kByte);

            tensor_image = tensor_image.permute({0, 3, 1, 2});
            tensor_image = tensor_image.toType(torch::kFloat);
            tensor_image = tensor_image.div(255);
            //tensor_image = tensor_image.to(torch::kCUDA);

            tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
            tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
            tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
            //auto img_var = torch::autograd::make_variable(tensor_image, false);

            inputs[i] = tensor_image[0];
        }

        auto t = (double) cv::getTickCount();
        torch::Tensor result = _module.forward({inputs}).toTensor();

        at::Tensor probs = torch::softmax(result, 1);
        at::Tensor preds = torch::argmax(result, 1);
        t = (double) cv::getTickCount() - t;
        // printf("execution time = %gs\n", t / cv::getTickFrequency());


        for(int i = 0; i < batch_size; i++)
        {
            auto pred = preds[i].item<int>();
            auto prob = probs[i][pred].item<float>();

            std::map<int, float> rslt;
            rslt[pred] = prob;
            ret.push_back(rslt);
        }

        std::cout << "predict: " << ret << "\t"
                  << "execute time: " << t / cv::getTickFrequency()
                  << std::endl;


//    for(int i = 0; i < ret.size(); i++)
//    {
//        std::map<int, float> rslt = ret[i];
//        std::cout << rslt << std::endl;
//    }
    }
    catch (...)
    {
        std::cout << "ERROR: predict runtime error!" << std::endl;
    }

    return ret;
}

maskface::~maskface()
{

}

