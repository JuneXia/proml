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


//void maskface::predict(cv::Mat images, std::vector<std::vector<int>> bboxes)
//{
//    std::cout << bboxes << std::endl;
//
//    cv::cvtColor(images, images, cv::COLOR_BGR2RGB);
//
//    std::vector<std::vector<int>>::iterator iter1;
//    std::vector<int> vec;
////    cv::Mat rois;
//    std::vector<cv::Mat> rois;
//
//    int margin = 0;
//
//    for(iter1 = bboxes.begin(); iter1 != bboxes.end(); iter1++)
//    {
//        cv::Mat roi;
//        vec = *iter1;
//        std::cout << vec << std::endl;
//
//        vec[0] = std::max(vec[0] - margin/2, 0);
//        vec[1] = std::max(vec[1] - margin/2, 0);
//        vec[2] = std::max(vec[2] + margin, 0);
//        vec[3] = std::max(vec[3] + margin, 0);
//
//        std::cout << vec << std::endl;
//        cv::Rect rect(vec[0], vec[1], vec[2], vec[3]);
//
////        cv::Mat tmp = images(rect);
////        std::cout << tmp.dims << std::endl;
////        std::cout << images.dims << std::endl;
////
////        std::cout << tmp << std::endl;
////        std::cout << tmp.data << std::endl;
////        std::cout << tmp.size << std::endl;
//
//        cv::resize(images(rect), roi, cv::Size(160, 160), 0, 0, cv::INTER_CUBIC);
//
//        cv::imshow("show", roi);
//        cv::waitKey();
//
//        rois.push_back(roi);
//    }
//
//
////    std::cout << rois << std::endl;
////    std::cout << rois.size() << std::endl;
////    std::cout << rois[0].rows << std::endl;
////    std::cout << rois[0].cols << std::endl;
//
//
//    std::vector<torch::jit::IValue> inputs;
//
////    std::cout << rois << std::endl;
////    std::cout << rois.data() << std::endl;
////
////    torch::Tensor tensor_image2 = torch::tensor(rois.data());
////    tensor_image2 = tensor_image2.permute({0, 3, 1, 2});
////    tensor_image2 = tensor_image2.toType(torch::kFloat);
////    tensor_image2 = tensor_image2.div(255);
////    torch::Tensor result2 = _module.forward(inputs).toTensor();
////    std::cout << result2 << std::endl;
//
//
//    torch::Tensor data = torch::zeros({rois.size(), 3, 160, 160});
//    data = data.toType(torch::kFloat);
////    std::cout << data << std::endl;
//
//    for(int i = 0; i < rois.size(); i++)
//    {
//        cv::imshow("show", rois[i]);
//        cv::waitKey();
//
//        std::cout << rois[i] << std::endl;
//        std::cout << rois[i].data << std::endl;
//        torch::Tensor tensor_image = torch::from_blob(rois[i].data, {1, rois[i].rows, rois[i].cols, 3}, torch::kByte);
//
//        std::cout << "tensor_image.sizes(): " << tensor_image.sizes() << std::endl;
//        std::cout << "tensor_image.element_size():" << tensor_image.element_size() << std::endl;
//        std::cout << "tensor_image.itemsize():" << tensor_image.itemsize() << std::endl;
//
//        tensor_image = tensor_image.permute({0, 3, 1, 2});
//        tensor_image = tensor_image.toType(torch::kFloat);
//        tensor_image = tensor_image.div(255);
//        //tensor_image = tensor_image.to(torch::kCUDA);
//
//        tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
//        tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
//        tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
//        //auto img_var = torch::autograd::make_variable(tensor_image, false);
//
//        std::cout << "tensor_image.sizes(): " << tensor_image.sizes() << std::endl;
//        std::cout << "tensor_image.element_size():" << tensor_image.element_size() << std::endl;
//        std::cout << "tensor_image.itemsize():" << tensor_image.itemsize() << std::endl;
//        std::cout << "tensor_image:" << tensor_image << std::endl;
//
//        std::cout << data[i] << std::endl;
//        data[i] = tensor_image[0];
//        std::cout << data[i] << std::endl;
//
//        std::cout << "data.sizes(): " << data[i].sizes() << std::endl;
//        std::cout << "data.element_size():" << data[i].element_size() << std::endl;
//        std::cout << "data.itemsize():" << data[i].itemsize() << std::endl;
//
//        inputs.emplace_back(data[i]);
//    }
//
//    std::cout << "data.sizes(): " << data.sizes() << std::endl;
//    std::cout << "data.element_size():" << data.element_size() << std::endl;
//    std::cout << "data.itemsize():" << data.itemsize() << std::endl;
//    std::cout << "data:" << data << std::endl;
//    std::cout << "data.data():" << data.data() << std::endl;
//
//    auto t = (double) cv::getTickCount();
//    //auto result = _module.forward({data});
//    torch::Tensor result = _module.forward({data}).toTensor();
//    std::cout << result << std::endl;
//    t = (double) cv::getTickCount() - t;
//    printf("execution time = %gs\n", t / cv::getTickFrequency());
//    inputs.pop_back();
//
//
//    std::cout << result << std::endl;
//
//    at::Tensor prob = torch::softmax(result,1);
//    at::Tensor prediction = torch::argmax(result, 1);
//
//
//    auto pp = prob.numpy_T();
//    std::cout << pp << std::endl;
//
//    auto pp2 = prediction[0].item<float>();
//    std::cout << pp2 << std::endl;
//
//    auto t1 = data.sizes();
//    std::cout << t1 << std::endl;
//
//    auto t2 = data.element_size();
//    std::cout << t2 << std::endl;
//
//    auto t3 = data.itemsize();
//    std::cout << t3 << std::endl;
//
//    for(int i = 0; i < prediction.sizes(); i++)
//    {
//
//    }
//
//
//
//    std::cout << prob << std::endl;
//    std::cout << prob[0] << std::endl;
//    std::cout << prediction << std::endl;
//    std::cout << prediction[0] << std::endl;
//
//    std::cout << prediction.item<float>() << std::endl;
//
//    auto results = prediction.sort(-1, true);
//    auto softmaxs = std::get<0>(results)[0].softmax(0);
//    auto indexs = std::get<1>(results)[0];
//
//    for (int i = 0; i < 3; ++i) {
//        auto idx = indexs[i].item<int>();
//        std::cout << "    ============= Top-" << i + 1
//                  << " =============" << std::endl;
//        //std::cout << "    Label:  " << labels[idx] << std::endl;
//        std::cout << "    With Probability:  "
//                  << softmaxs[i].item<float>() * 100.0f << "%" << std::endl;
//    }
//
////    auto foo = torch::maybe_data_ptr(prediction);
////
////    //auto foo = prediction.accessor<int, 2>();
////
////    for(int i = 0; i < foo.size(0); i++)
////    {
////        std::cout << prediction[i] << std::endl;
////        std::cout << foo[i][i] << std::endl;
////    }
//
//    std::vector<std::map<int, int>> ret;
////    std::map<int, int> rslt()
////    ret.push_back()
//
//
////    for(int i = 0; i < rois.size(); i++)
////    {
////        std::cout << rois[i] << std::endl;
////        std::cout << rois[i].data << std::endl;
////        torch::Tensor tensor_image = torch::from_blob(rois[i].data, {1, rois[i].rows, rois[i].cols, 3}, torch::kByte);
////
////        std::cout << "tensor_image.sizes(): " << tensor_image.sizes() << std::endl;
////        std::cout << "tensor_image.element_size():" << tensor_image.element_size() << std::endl;
////        std::cout << "tensor_image.itemsize():" << tensor_image.itemsize() << std::endl;
////
////        tensor_image = tensor_image.permute({0, 3, 1, 2});
////        tensor_image = tensor_image.toType(torch::kFloat);
////        tensor_image = tensor_image.div(255);
////        //tensor_image = tensor_image.to(torch::kCUDA);
////
////        tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
////        tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
////        tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
////        //auto img_var = torch::autograd::make_variable(tensor_image, false);
////
////        inputs.emplace_back(tensor_image);
////    }
////
////    auto t = (double) cv::getTickCount();
////    torch::Tensor result = _module.forward(inputs).toTensor();
////    std::cout << result << std::endl;
////    t = (double) cv::getTickCount() - t;
////    printf("execution time = %gs\n", t / cv::getTickFrequency());
////    inputs.pop_back();
////
////
////    std::cout << result << std::endl;
////
////    at::Tensor prob = torch::softmax(result,1);
////    auto prediction = torch::argmax(result, 1);
////
////    std::cout << prob << std::endl;
////    std::cout << prediction << std::endl;
//}


std::vector<std::map<int, float>> maskface::predict(cv::Mat images, std::vector<std::vector<int>> bboxes)
{
    if(images.empty())
    {
        std::cout << "ERROR: images empty, please check your image!" << std::endl;
        throw images;
    }
    if(bboxes.size() == 0 or bboxes[0].size() != 4)
    {
        std::cout << "ERROR: bboxes empty, please check your bboxes!" << std::endl;
        throw bboxes;
    }

    std::vector<std::map<int, float>> ret;
    try
    {
        std::vector<std::vector<int>>::iterator iter1;
        std::vector<int> vec;
        std::vector<cv::Mat> rois;

        // cv::cvtColor(images, images, cv::COLOR_BGR2RGB);

        auto batch_size = bboxes.size();

        for(iter1 = bboxes.begin(); iter1 != bboxes.end(); iter1++)
        {
            cv::Mat roi;
            vec = *iter1;
            //std::cout << vec << std::endl;

            vec[0] = std::max(vec[0] - _margin/2, 0);
            vec[1] = std::max(vec[1] - _margin/2, 0);
            vec[2] = std::max(vec[2] + _margin, 0);
            vec[3] = std::max(vec[3] + _margin, 0);

            //std::cout << vec << std::endl;
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

