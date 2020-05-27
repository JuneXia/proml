#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "inc/gc_face_sdk.h"
#include "inc/merror.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image> \n";
        return -1;
    }
    char *model_path = (char *)argv[1];
    char *image_path = (char *)argv[2];

    MRESULT ret = MOK;
    MHandle handle;
    ret = GCAInitEngine(model_path, &handle);
    if (ret != MOK)
    {
        printf("GCAInitEngine fail: %d\n", ret);
        return -1;
    }
    else
    {
        printf("GCAInitEngine success: %d\n", ret);
    }
    
    try
    {
        int count = 0;
        while(count < 10)
        {
            count ++;
            // MRECT faceRect = {30, 50, 70, 70};  // 需要您指定人脸矩形框
            // MRECT faceRect = {295, 225, 80, 120};
            MRECT faceRect[2] = { {295, 225, 80, 120}, {576, 80, 120, 170} };
            GCAI_ClassInfo classInfo;
            LPGCAI_ClassInfo pClassInfo = new GCAI_ClassInfo[2];
            ret = GCAIPredict(handle, image_path, faceRect, 2, pClassInfo);
            if (ret != MOK)
            {
                printf("GCAIPredict fail: %d\n", ret);
            }
            else
            {
                cv::Mat image = cv::imread(image_path);
                for (int i=0; i<2; i++)
                {
                    // std::cout << (pClassInfo+i)->confidence << ", " << (pClassInfo+i)->label << std::endl;
                    cv::Scalar color;
                    if ((pClassInfo+i)->label == 0)
                    {
                        color = cv::Scalar(0, 0, 255);
                    }
                    else if ((pClassInfo+i)->label == 1)
                    {
                        color = cv::Scalar(0, 255, 0);
                    }
                    cv::Rect rect(faceRect[i].left, faceRect[i].top, faceRect[i].width, faceRect[i].height);
                    cv::rectangle(image, rect, color, 2);
                }
                cv::imshow("show", image);
                cv::waitKey(100);
            }
        }
    }
    catch (...)
    {
        std::cout << "catch" << std::endl;
    }

    GCAIUninitEngine(handle);
    return 0;
}
