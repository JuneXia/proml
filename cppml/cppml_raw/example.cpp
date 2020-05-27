//
// Created by xiajun on 20-3-17.
//

#include "maskface.h"
#include <memory>
#include <string>
#include <vector>


int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image>  <path-to-category-text>\n";
        // return -1;
    }
    const char *model_path = "/home/xiajun/dev/proml/maskface/mobilenetv1_conv1x1_in96_netclip_eph112.pt";
    //const char *image_path = "/home/xiajun/res/face/maskface/Experiment/test-images/test_00000002.png";
    const char *image_path = "/home/xiajun/res/face/maskface/test-images/test_00000008.jpg";

    maskface module = maskface(model_path);

//    cv::namedWindow("show", cv::WINDOW_AUTOSIZE);
    cv::Mat input;
    cv::Mat image = cv::imread(image_path, 1);

    std::vector<std::vector<int>> bboxes = { {30, 50, 70, 70}/*, {70, 10, 70, 70}*/ };

    try
    {
        auto result = module.predict(image, bboxes);
        std::cout << result << std::endl;
        while(true)
        {
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
            result = module.predict(image, bboxes);
        }
    }
    catch (...)
    {
        std::cout << "catch" << std::endl;
    }

    return 0;
}

int main_videostream(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image>  <path-to-category-text>\n";
        // return -1;
    }
    const char *model_path = "/home/xiajun/dev/proml/maskface/mobilenetv1_conv1x1_in96_netclip_eph112.pt";
    //const char *image_path = "/home/xiajun/res/face/maskface/Experiment/test-images/test_00000002.png";
    const char *image_path = "/home/xiajun/res/face/maskface/test-images/test_00000008.jpg";

    maskface module = maskface(model_path);

//    cv::namedWindow("show", cv::WINDOW_AUTOSIZE);
    cv::Mat input;
    //cv::Mat image = cv::imread(image_path, 1);
    //auto capture = cv::VideoCapture(0);
    auto capture = cv::VideoCapture("rtsp://admin:admin888@10.10.2.132:554/h264/ch1/main/av_stream");

    std::vector<std::vector<int>> bboxes = { {100, 100, 120, 120}, {150, 150, 120, 120}, {200, 200, 120, 120} };

    try
    {
        int count = 0;
        while (capture.isOpened())
        {
            count++;
            cv::Mat image;
            //capture.read(image);
            capture >> image;
            if (image.empty())
            {
                std::cout << "capture read empty" << std::endl;
                cv::waitKey(50);
                continue;
            }

            if (count % 20 == 0)
            {
                std::cout << "count: " << count << std::endl;
                auto result = module.predict(image, bboxes);
                std::cout << result << std::endl;

                for (int i = 0; i < result.size(); i++)
                {
                    cv::Scalar color;
                    std::map<int, float>::iterator iter = result[i].begin();

                    int label = iter->first;
                    float confidence = iter->second;
                    if (label == 0)  // 未戴口罩
                        color = cv::Scalar(0, 0, 255);
                    else if (label == 1)  // 戴口罩
                        color = cv::Scalar(0, 255, 0);
                    else
                        color = cv::Scalar(255, 0, 0);  // 暂时不存在这种情况

                    cv::rectangle(image, cv::Rect(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]), color);
                    std::stringstream text;
                    text << confidence;
                    cv::putText(image, text.str(), cv::Point(bboxes[i][0] + 5, bboxes[i][1] + 15), cv::FONT_HERSHEY_PLAIN, 1.0, color);
                }
            }
            cv::imshow("show", image);
            cv::waitKey(10);
        }
    }
    catch (...)
    {
        std::cout << "catch" << std::endl;
    }

    return 0;
}