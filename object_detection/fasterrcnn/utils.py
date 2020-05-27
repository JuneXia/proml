#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:36:23 2018

@author: jon-liu
"""

import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
import cv2
import keras.engine as KE


def anchor_gen(featureMap_size, ratios, scales, rpn_stride, anchor_stride):
    ratios, scales = np.meshgrid(ratios, scales)
    ratios, scales = ratios.flatten(), scales.flatten()

    width = scales / np.sqrt(ratios)
    height = scales * np.sqrt(ratios)

    shift_x = np.arange(0, featureMap_size[0], anchor_stride) * rpn_stride
    shift_y = np.arange(0, featureMap_size[1], anchor_stride) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    centerX, anchorX = np.meshgrid(shift_x, width)
    centerY, anchorY = np.meshgrid(shift_y, height)
    boxCenter = np.stack([centerY, centerX], axis=2).reshape(-1, 2)
    boxSize = np.stack([anchorX, anchorY], axis=2).reshape(-1, 2)

    boxes = np.concatenate([boxCenter - 0.5 * boxSize, boxCenter + 0.5 * boxSize], axis=1)
    return boxes


def non_max_suppression(boxes, scores, nms_threshold):
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]

    areas = (y2 - y1) * (x2 - x1)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        ix = idxs[0]
        ious = compute_iou(boxes[ix], boxes[idxs[1:]], areas[ix], areas[idxs[1:]])
        keep.append(ix)
        remove_idxs = np.where(ious > nms_threshold)[0] + 1
        idxs = np.delete(idxs, remove_idxs)
        idxs = np.delete(idxs, 0)
    return np.array(keep, dtype=np.int32)


def compute_iou(box, boxes, area, areas):
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    interSec = np.maximum(y2 - y1, 0) * np.maximum(x2 - x1, 0)
    union = areas[:] + area - interSec
    iou = interSec / union
    return iou


def compute_overlap(boxes1, boxes2):
    areas1 = (boxes1[:, 3] - boxes1[:, 1]) * (boxes1[:, 2] - boxes1[:, 0])
    areas2 = (boxes2[:, 3] - boxes2[:, 1]) * (boxes2[:, 2] - boxes2[:, 0])
    overlap = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        overlap[:, i] = compute_iou(box, boxes1, areas2[i], areas1)
    return overlap


def build_rpnTarget(boxes, anchors, config):
    """
    :param boxes: 要回归的目标框，即 ground_truth
    :param anchors: 事先计算好的anchors
    :param config:
    :return: rpn_match: anchors label, rpn_bboxes rpn要回归的bbox偏移量
    """
    rpn_match = np.zeros(anchors.shape[0], dtype=np.int32)
    rpn_bboxes = np.zeros((config.train_rois_num, 4))

    # step1: 计算 anchors 和 每个 boxes 的 iou
    iou = compute_overlap(anchors, boxes)

    # step2: 提取每个anchor框与boxes的iou最大的那一个
    maxArg_iou = np.argmax(iou, axis=1)
    max_iou = iou[np.arange(iou.shape[0]), maxArg_iou]

    # step3: 计算正负样本索引
    postive_anchor_idxs = np.where(max_iou > 0.4)[0]
    negative_anchor_idxs = np.where(max_iou < 0.1)[0]

    # step4: 将正样本下标置为1，负样本下标置为-1
    rpn_match[postive_anchor_idxs] = 1
    rpn_match[negative_anchor_idxs] = -1

    # step5: 这一步我也很迷 TODO: DEBUG, (paper中有提到产生proposal的两种方法之一)
    maxIou_anchors = np.argmax(iou, axis=0)
    rpn_match[maxIou_anchors] = 1

    # step6: 如果正样本数量超过训练roi总数的1/2，也就是此时正样本数量太多了，则要随机剔除一些正样本
    ids = np.where(rpn_match == 1)[0]  # 所有正样本 index
    # 随机剔除一些正样本 = 正样本总量 - 训练roi总数的1/2
    extral = len(ids) - config.train_rois_num // 2
    if extral > 0:
        # 随机剔除一些正样本
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0

    # step7: if (负样本总量 - (训练roi总数 - 正样本总量) > 0)，也就是此时负样本数量太多了，则要随机剔除一些负样本
    ids = np.where(rpn_match == -1)[0]  # 所有负样本 index
    # 一张图片最终只取train_rois_num个侯选框
    # 随机剔除一些负样本 = 负样本总量 - (训练roi总数 - 正样本总量)
    extral = len(ids) - (config.train_rois_num - np.where(rpn_match == 1)[0].shape[0])
    if extral > 0:
        # 随机剔除一些负样本
        ids_ = np.random.choice(ids, extral, replace=False)
        rpn_match[ids_] = 0

    # step8: anchor框相对ground_truth的偏移量
    idxs = np.where(rpn_match == 1)[0]
    ix = 0
    for i, a in zip(idxs, anchors[idxs]):
        # 取出ground_truth box
        gt = boxes[maxArg_iou[i]]

        # 计算ground_truth的中心点及宽高
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_centy = gt[0] + 0.5 * gt_h
        gt_centx = gt[1] + 0.5 * gt_w

        # 计算anchor框的中心点和宽高
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_centy = a[0] + 0.5 * a_h
        a_centx = a[1] + 0.5 * a_w

        # 计算rpn要回归的偏移量
        rpn_bboxes[ix] = [(gt_centy - a_centy) / a_h, (gt_centx - a_centx) / a_w,
                          np.log(gt_h / a_h), np.log(gt_w / a_w)]
        rpn_bboxes[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1
    return rpn_match, rpn_bboxes


def batch_slice(inputs, graph_fn, batch_size, names=None):
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)

    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]
    return result


class shapeData():
    def __init__(self, image_size, config):
        self.image_size = image_size
        #        self.num_image = num_image
        self.config = config

        self.debug_count = 0

    def load_data(self):
        images = np.zeros((self.image_size[0], self.image_size[1], 3))
        #        bboxs = []
        #        ids = []
        #        rpn_match = []
        #        rpn_bboxes = []
        anchors = anchor_gen(self.config.featureMap_size, self.config.ratios, self.config.scales,
                             self.config.rpn_stride, self.config.anchor_stride)

        images, bboxs, ids = self.random_image(self.image_size)
        rpn_match, rpn_bboxes = build_rpnTarget(bboxs, anchors, self.config)
        return images, bboxs, ids, rpn_match, rpn_bboxes, anchors

    def random_image(self, image_size):
        typeDict = {'square': 1, 'circle': 2, 'triangle': 3}
        H, W = image_size[0], image_size[1]
        # image = np.random.randn(H, W, 3)
        red = np.ones((64, 64, 1)) * 30
        green = np.ones((64, 64, 1)) * 60
        blue = np.ones((64, 64, 1)) * 90
        image = np.concatenate([red, green, blue], axis=2)
        num_obj = random.sample([1, 2, 3, 4], 1)[0]
        # num_obj = 1
        bboxs = np.zeros((num_obj, 4))
        Ids = np.zeros((num_obj, 1))
        shapes = []
        dims = np.zeros((num_obj, 3))
        for i in range(num_obj):
            shape = random.sample(list(typeDict), 1)[0]
            shapes.append(shape)

            Ids[i] = typeDict[shape]
            x, y = np.random.randint(H // 4, W // 4 + W // 2, 1)[0], np.random.randint(H // 4, W // 4 + W // 2, 1)[0]
            # x, y = 32, 64
            s = np.random.randint(H // 16, W // 8, 1)[0]
            # s = 12
            dim = x, y, s
            dims[i] = dim
            # color = random.randint(1, 255)
            # if type(color) is float or type(x) is float or type(s) is float:
            #     print('debug')
            # image = self.draw_shape(image, shape, dim, color)
            bboxs[i] = self.draw_boxes(dim)

            # cv2.imshow('show', image)
            # cv2.waitKey()

        if False:
            image_copy = image.copy()
            for j in range(bboxs.shape[0]):
                color = random.randint(1, 255)
                shape = shapes[j]
                dim = dims[j]
                image_copy = self.draw_shape(image_copy, shape, dim, color)
                bbox = bboxs[j].astype(np.int32)
                cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
            cv2.imshow('no-nms', image_copy)
            cv2.waitKey()


        keep_idxs = non_max_suppression(bboxs, np.arange(num_obj), 0.01)
        bboxs = bboxs[keep_idxs]
        Ids = Ids[keep_idxs]
        shapes = [shapes[i] for i in keep_idxs]
        dims = dims[keep_idxs]

        for j in range(bboxs.shape[0]):
            color = (random.randint(1, 255), random.randint(1, 255), random.randint(1, 255))
            shape = shapes[j]
            dim = dims[j]
            image = self.draw_shape(image, shape, dim, color)
            # bbox = bboxs[j].astype(np.int32)
            # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
        if False:
            cv2.imshow('nms', image)
            cv2.waitKey()

        cv2.imwrite(str(self.debug_count) + '.jpg', image)
        self.debug_count += 1
        return image, bboxs, Ids

    def draw_shape(self, image, shape, dims, color):
        x, y, s = dims.astype(np.int32)
        try:
            if shape == 'square':
                cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
            elif shape == "circle":
                cv2.circle(image, (x, y), s, color, -1)
            elif shape == "triangle":
                points = np.array([[(x, y - s),
                                    (x - s / math.sin(math.radians(60)), y + s),
                                    (x + s / math.sin(math.radians(60)), y + s),
                                    ]], dtype=np.int32)
                cv2.fillPoly(image, points, color)
        except Exception as err:
            print('debug', err)

        return image

    def draw_boxes(self, dims):
        x, y, s = dims
        bbox = [x - s, y - s, x + s, y + s]
        bbox = np.array(bbox)
        return bbox







