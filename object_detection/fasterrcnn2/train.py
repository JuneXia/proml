# -*- coding: utf-8 -*-
import os
import time
import torch.nn as nn
import torch
import random
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import torch.nn.functional as F
from tutorial.tools.common_tools import set_seed
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.models.mobilenet import mobilenet_v2
from nets.rpn import RegionProposalNetwork as RPN

from datasets import datasets
from losses import rpn_loss
from utils import anchors as Anchors

set_seed(1)  # 设置随机种子

BASE_DIR = "/home/xiajun/res/person"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# classes_coco
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def vis_bbox(img, output, classes, max_vis=40, prob_thres=0.4):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img, aspect='equal')

    out_boxes = output_dict["boxes"].cpu()
    out_scores = output_dict["scores"].cpu()
    out_labels = output_dict["labels"].cpu()

    num_boxes = out_boxes.shape[0]
    for idx in range(0, min(num_boxes, max_vis)):

        score = out_scores[idx].numpy()
        bbox = out_boxes[idx].numpy()
        class_name = classes[out_labels[idx]]

        if score < prob_thres:
            continue

        ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                   edgecolor='red', linewidth=3.5))
        ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')
    plt.show()
    plt.close()


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target



if __name__ == "__main__1":  # RPN 测试
    net = mobilenet_v2(num_classes=2, width_mult=0.35, inverted_residual_setting=None, round_nearest=8).to(device)
    rpn_net = RPN(1280, 512, 9)

    inputs = torch.zeros((2, 3, 128, 128))
    # cls = net(inputs)
    feature_map = net.features(inputs)
    rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map)


def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    # targets_ = tuple()
    # for t in targets:
    #     target = dict()
    #     for k, v in t.items():
    #         if v is None:
    #             target[k] = v
    #         else:
    #             target[k] = torch.tensor(v)
    #     targets_ += (target, )
    return torch.stack(images, 0), targets


if __name__ == "__main__":
    LR = 0.001
    num_classes = 2
    batch_size = 2
    start_epoch, max_epoch = 0, 30
    train_dir = "/home/xiajun/dev/proml/object_detection/fasterrcnn2/VOCdevkit/VOC2007_train.txt"
    train_transform = Compose([ToTensor(), RandomHorizontalFlip(0.5)])

    # step 1: data
    train_set = datasets.VOC2007Dataset(vocfile=train_dir, transforms=train_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn)

    base_size = 16
    ratios = [0.5, 1, 2]
    anchor_scales = [8, 16, 32]
    anchors_gen = Anchors.AnchorTargetCreator()
    anchor_base = Anchors.generate_anchor_base(base_size=base_size, anchor_scales=anchor_scales, ratios=ratios)









    net = mobilenet_v2(num_classes=2, width_mult=0.35, inverted_residual_setting=None, round_nearest=8).to(device)
    rpn_net = RPN(1280, 512, 9).to(device)

    # inputs = torch.zeros((2, 3, 128, 128))
    # cls = net(inputs)
    # feature_map = net.features(inputs)
    # rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map)

    classify_loss = rpn_loss.ClassifyLoss()

    for data in train_loader:
        images, targets = data
        images = images.to(device)

        targets_ = list()
        for t in targets:
            target = dict()
            for k, v in t.items():
                if v is None:
                    target[k] = v
                else:
                    target[k] = v.to(device)
            targets_ += (target,)
        targets = targets_

        feature_map = net.features(images)



        # 提取特征图宽高
        feature_width = feature_map.shape[2]
        feature_height = feature_map.shape[3]

        # 计算在特征图上滑动anchor_base时的跨度
        # 一副图片经过backbone后得到的是多次下采样后的特征图，anchor框是指在输入图片上的anchor框.
        # 而要想在输入图片上均匀生成等间距的anchor_base，则需要有一个合理的anchor_base间隔，也就是下面即将要计算的feature_stride
        remainder = 1 if image_size[0] % feature_width > 0 else 0
        feature_stride = image_size[0] // feature_width + remainder

        anchor = Anchors.enumerate_shifted_anchor(np.array(anchor_base), feature_stride, feature_height, feature_width)

        rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map)

        anchors_gen(target['boxes'], anchor, img_size)

        classify_loss(rpn_class, targets['label'])


        classify_loss()

    # step 2: model
    net = mobilenet_v2(num_classes=2, width_mult=0.35, inverted_residual_setting=None, round_nearest=8).to(device)
    rpn_net = RPN(1280, 512, 9)

    inputs = torch.zeros((2, 3, 128, 128))
    # cls = net(inputs)
    feature_map = net.features(inputs)
    rpn_class, rpn_prob, rpn_bbox = rpn_net(feature_map)




    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes)  # replace the pre-trained head with a new one

    model.to(device)

    # step 3: loss
    # in lib/python3.6/site-packages/torchvision/models/detection/roi_heads.py
    # def fastrcnn_loss(class_logits, box_regression, labels, regression_targets)

    # step 4: optimizer scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # step 5: Iteration

    for epoch in range(start_epoch, max_epoch):

        model.train()
        for iter, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # if torch.cuda.is_available():
            #     images, targets = images.to(device), targets.to(device)

            loss_dict = model(images, targets)  # images is list; targets is [ dict["boxes":**, "labels":**], dict[] ]

            losses = sum(loss for loss in loss_dict.values())

            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} ".format(
                epoch, max_epoch, iter + 1, len(train_loader), losses.item()))

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

    # test
    model.eval()

    # config
    vis_num = 5
    vis_dir = os.path.join(BASE_DIR, "PennFudanPed", "PNGImages")
    img_names = list(filter(lambda x: x.endswith(".png"), os.listdir(vis_dir)))
    random.shuffle(img_names)
    preprocess = transforms.Compose([transforms.ToTensor(), ])

    for i in range(0, vis_num):

        path_img = os.path.join(vis_dir, img_names[i])
        # preprocess
        input_image = Image.open(path_img).convert("RGB")
        img_chw = preprocess(input_image)

        # to device
        if torch.cuda.is_available():
            img_chw = img_chw.to('cuda')
            model.to('cuda')

        # forward
        input_list = [img_chw]
        with torch.no_grad():
            tic = time.time()
            print("input img tensor shape:{}".format(input_list[0].shape))
            output_list = model(input_list)
            output_dict = output_list[0]
            print("pass: {:.3f}s".format(time.time() - tic))

        # visualization
        vis_bbox(input_image, output_dict, COCO_INSTANCE_CATEGORY_NAMES, max_vis=20,
                 prob_thres=0.5)  # for 2 epoch for nms


