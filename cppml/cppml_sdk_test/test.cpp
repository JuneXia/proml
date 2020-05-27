#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "inc/gc_face_sdk.h"
#include "inc/merror.h"

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        std::cerr << "usage: CppProject <path-to-exported-script-module> "
                  << "<path-to-image>  <path-to-category-text>\n";
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
        

    // cv::namedWindow("show", cv::WINDOW_AUTOSIZE);
    // cv::Mat input;
    // cv::Mat image = cv::imread(image_path, 1);

    // std::vector<std::vector<int>> bboxes = { {30, 50, 70, 70}/*, {70, 10, 70, 70}*/ };

    try
    {
        int count = 0;
        while(count < 10)
        {
            count ++;
            MRECT faceRect = {30, 50, 70, 70};
            GCAI_ClassInfo classInfo;
            ret = GCAIPredict(handle, image_path, &faceRect, &classInfo);
            if (ret != MOK)
            {
                printf("GCAIPredict fail: %d\n", ret);
            }
            else
            {
                std::cout << classInfo.confidence << ", " << classInfo.label << std::endl;
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
