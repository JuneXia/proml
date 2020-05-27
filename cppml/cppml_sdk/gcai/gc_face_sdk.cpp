#include "maskface.h"
#include "gc_face_sdk.h"
#include "merror.h"

MRESULT GCAInitEngine(MPChar modelPath, MHandle* hEngine)
{
    maskface *module = new maskface(modelPath);
    *hEngine = (MHandle *)module;
    return 0;
}

MRESULT GCAIPredict(MHandle hEngine, MPChar imagePath, PMRECT faceRect, MUInt16 numFace, LPGCAI_ClassInfo classInfo)
{
    maskface *handle = (maskface *)hEngine;

    cv::Mat image = cv::imread(imagePath, 1);
    
    std::vector<std::vector<int>> rects;
    for (int i=0; i<numFace; i++)
    {
        std::vector<MInt32> rect{faceRect[i].left, faceRect[i].top, faceRect[i].width, faceRect[i].height};
        rects.push_back(rect);
    }
    
    std::vector<std::map<int, float>> result = handle->predict(image, rects);

    for(int i = 0; i < result.size(); i++)
    {
        std::map<int, float> rslt = result[i];
        std::map<int, float>::iterator iter = result[i].begin();

        (classInfo + i)->label = iter->first;
        (classInfo + i)->confidence = iter->second;
    }

    return 0;
}

MRESULT GCAIUninitEngine(MHandle hEngine)
{
    maskface *handle = (maskface *)hEngine;
    delete handle;
    return 0;
}
