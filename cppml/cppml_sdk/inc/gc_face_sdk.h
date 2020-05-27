/*******************************************************************************
* Copyright(c) GCRobot, All right reserved.
*
* This file is GCRobot's property. It contains GCRobot's trade secret, proprietary
* and confidential information.
*
* DO NOT DISTRIBUTE, DO NOT DUPLICATE OR TRANSMIT IN ANY FORM WITHOUT PROPER
* AUTHORIZATION.
*
* If you are not an intended recipient of this file, you must not copy,
* distribute, modify, or take any action in reliance on it.
*
* If you have received this file in error, please immediately notify GCRobot and
* permanently delete the original and any copy of any file and any printout
* thereof.
*********************************************************************************/

#ifndef _GC_SDK_ASF_H_
#define _GC_SDK_ASF_H_

#include "amcomdef.h"
#include "asvloffscreen.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct {
    	MUInt32 label;
		MFloat  confidence;
	}GCAI_ClassInfo, *LPGCAI_ClassInfo;

	/************************************************************************
	* 初始化引擎
	************************************************************************/
	MRESULT GCAInitEngine(
		MPChar modelPath, // [in]  模型存储路径
		MHandle* hEngine  // [out] 初始化返回的引擎handle
		);
	
	/************************************************************************
	* 是否待口罩预测接口。
	************************************************************************/
	MRESULT GCAIPredict(
		MHandle hEngine,             // [in] 引擎handle
		MPChar imagePath,            // [in] 图片路径
		PMRECT faceRect,             // [in] 人脸矩形框
		MUInt16 numFace,             // [in] 有多少个人脸框
		LPGCAI_ClassInfo classInfo   // [out] 人脸类别信息
		);

	/************************************************************************
	* 销毁引擎
	************************************************************************/
	MRESULT GCAIUninitEngine(MHandle hEngine);

#ifdef __cplusplus
}
#endif
#endif
