import os
import cv2
import imageio
import numpy as np
from torchvision.models.mobilenet import mobilenet_v2
import torch
import anchors as Anchors
from libml.utils.config import SysConfig

if __name__ == '__main__':  # 演示一组anchors中的9个anchor框在输入图片中的尺度
    net = mobilenet_v2()

    base_size = 16  # anchor基础尺寸，当base_size=16时，anchor最小长或宽可低至 16*8*sqrt(0.5), 即16*8/sqrt(2)
                    # 如果希望对检测小尺寸目标有利，则应该要设置更小的 base_size，或者调低anchor_scales中的尺度。
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]

    # 生成一组anchor基，这组anchor基此时还没有中心点，它们只是对应着9种尺度。
    anchor_base = Anchors.generate_anchor_base(base_size=base_size, anchor_scales=anchor_scales, ratios=ratios)
    image_path = os.path.join(SysConfig["home_path"], SysConfig["proml_path"], "data/voc2007-train-000030.jpg")
    bboxes = np.array([[51, 225, 241, 474], [382, 335, 558, 472], [419, 261, 540, 477], [280, 220, 360, 300]])

    image = cv2.imread(image_path)
    image_size = (image.shape[0], image.shape[1])

    for box in bboxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('ground_truth', image)
    cv2.waitKey()

    # 作为示例，这里选择原图像的中心点作为anchor基的中心点，并画出一组anchor作为示例：
    anchor_center = np.array([image_size[0] / 2, image_size[1] / 2, image_size[0] / 2, image_size[1] / 2])

    # anchor_base + anchor_center 后就得到了这组anchor的坐标了
    anchors = (anchor_base + anchor_center).astype(np.int32)

    # 显示画框示例
    images = []
    for ach in anchors:
        cv2.rectangle(image, (ach[0], ach[1]), (ach[2], ach[3]), (180, 0, 0), 2)
        images.append(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
        cv2.imshow('anchor_example', image)
        cv2.waitKey()

    imageio.mimsave('./anchor_example.gif', images, 'GIF', duration=0.6)
    print('finish')


if __name__ == '__main__':  # 演示如何从整张图片中的所有anchor框中提取与ground-truth有关的anchor框
    net = mobilenet_v2()

    base_size = 16  # anchor基础尺寸，当base_size=16时，anchor最小长或宽可低至 16*8*sqrt(0.5), 即16*8/sqrt(2)
                    # 如果希望对检测小尺寸目标有利，则应该要设置更小的 base_size，或者调低anchor_scales中的尺度。
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]


    # step1: 生成anchor_base
    # ===========================================================
    # 生成一组anchor基，这组anchor基此时还没有中心点，它们只是对应着9种尺度。
    anchor_base = Anchors.generate_anchor_base(base_size=base_size, anchor_scales=anchor_scales, ratios=ratios)
    image_path = os.path.join(SysConfig["home_path"], SysConfig["proml_path"], "data/voc2007-train-000030.jpg")
    bboxes = np.array([[ 51, 225, 241, 474], [382, 335, 558, 472], [419, 261, 540, 477], [280, 220, 360, 300]])

    image = cv2.imread(image_path)
    image_size = (image.shape[0], image.shape[1])


    # step2: 计算backbone得到的特征图
    # ===========================================================
    # 由输入图片计算特征图
    t = np.expand_dims(image, axis=0).astype(np.float32)
    t = torch.tensor(t)
    t = t.permute((0, 3, 1, 2))
    features = net.features(t)

    # 提取特征图宽高
    feature_width = features.shape[2]
    feature_height = features.shape[3]

    # 计算在特征图上滑动anchor_base时的跨度
    # 一副图片经过backbone后得到的是多次下采样后的特征图，anchor框是指在输入图片上的anchor框.
    # 而要想在输入图片上均匀生成等间距的anchor_base，则需要有一个合理的anchor_base间隔，也就是下面即将要计算的feature_stride
    remainder = 1 if image_size[0] % feature_width > 0 else 0
    feature_stride = image_size[0] // feature_width + remainder


    # step3: 根据特征图尺寸以及anchor_base生成正张输入图片上的anchors
    # ===========================================================
    # 生成所有的先验框：根据特征图宽、高和跨度对 anchor_base 进行平移，使其铺满至整副图片
    anchor = Anchors.enumerate_shifted_anchor(np.array(anchor_base), feature_stride, feature_height, feature_width)


    # step4: 通过nms从所有的anchor框提取和ground-truth有关的anchor框标签
    # ===========================================================
    # 上面得到的是输入图片上的所有anchor框，但实际上这些anchor框中大部分都不是我们想要的，
    # 我们想要的只是这些anchor框中和ground-truth有关的部分。
    # 通过nms从所有的anchor框提取和ground-truth有关的anchor框标签
    anchor_target_creator = Anchors.AnchorTargetCreator()
    argmax_ious, label = anchor_target_creator._create_label(anchor, bboxes)


    # 至此，所有的anchor框已经生成完毕，下面是可视化部分
    # ===========================================================
    # 画出ground-truth框
    for box in bboxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow('ground-truth', image)
    cv2.waitKey()

    # 画出最终的anchor框(由于负样本太多，这里只画出了和正样本相关的anchor框)
    images = []
    anchor = anchor[np.where(label == 1)]
    for ach in anchor:
        cv2.rectangle(image, (ach[0], ach[1]), (ach[2], ach[3]), (0, 0, 200), 2)
        images.append(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
        cv2.imshow('pos-anchor', image)
        cv2.waitKey()

    imageio.mimsave('./pos-anchor.gif', images, 'GIF', duration=0.6)

    print('finish')
