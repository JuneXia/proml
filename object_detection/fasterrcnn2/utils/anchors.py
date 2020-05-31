# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# 参考网易云课堂 > 利用keras实现fasterrcnn-刘镇硕
# 不推荐，建议废弃
import numpy as np


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


def compute_iou(box, boxes, area, areas):
    y1 = np.maximum(box[0], boxes[:, 0])
    x1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    interSec = np.maximum(y2-y1, 0) * np.maximum(x2-x1, 0)
    union = areas[:] + area - interSec
    iou = interSec / union
    return iou


def compute_overlap(boxes1, boxes2):
    areas1 = (boxes1[:,3] - boxes1[:,1]) * (boxes1[:,2] - boxes1[:,0])
    areas2 = (boxes2[:,3] - boxes2[:,1]) * (boxes2[:,2] - boxes2[:,0])
    overlap = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes2.shape[0]):
        box = boxes2[i]
        overlap[:,i] = compute_iou(box, boxes1, areas2[i], areas1)
    return overlap


class AnchorsGenerator(object):
    def __init__(self, n_sample=256):
        self.rpn_bbox_std_dev = np.array([0.1, 0.1, 0.2, 0.2])
        self.n_sampel = n_sample

    def __call__(self, feature_map_size, gt_bboxes):
        # feature_map_size = [8, 8]
        scales = [4, 8, 16]
        ratios = [0.5, 1, 2]
        rpn_stride = 8
        anchor_stride = 1
        anchors = anchor_gen(feature_map_size, ratios, scales, rpn_stride, anchor_stride)
        rpn_match, rpn_bboxes = self.build_rpn_target(gt_bboxes, anchors)


    def build_rpn_target(self, gt_bboxes, anchors):
        rpn_match = np.zeros(anchors.shape[0], dtype=np.int32)
        rpn_bboxes = np.zeros((self.n_sampel, 4))

        iou = compute_overlap(anchors, gt_bboxes)
        maxArg_iou = np.argmax(iou, axis=1)
        max_iou = iou[np.arange(iou.shape[0]), maxArg_iou]
        postive_anchor_idxs = np.where(max_iou > 0.4)[0]
        negative_anchor_idxs = np.where(max_iou < 0.1)[0]

        rpn_match[postive_anchor_idxs] = 1
        rpn_match[negative_anchor_idxs] = -1
        maxIou_anchors = np.argmax(iou, axis=0)
        rpn_match[maxIou_anchors] = 1

        ids = np.where(rpn_match == 1)[0]
        extral = len(ids) - self.n_sampel // 2
        if extral > 0:
            ids_ = np.random.choice(ids, extral, replace=False)
            rpn_match[ids_] = 0

        ids = np.where(rpn_match == -1)[0]
        extral = len(ids) - (self.n_sampel - np.where(rpn_match == 1)[0].shape[0])
        if extral > 0:
            ids_ = np.random.choice(ids, extral, replace=False)
            rpn_match[ids_] = 0

        idxs = np.where(rpn_match == 1)[0]
        ix = 0
        for i, a in zip(idxs, anchors[idxs]):
            gt = gt_bboxes[maxArg_iou[i]]

            gt_h = gt[2] - gt[0]
            gt_w = gt[3] - gt[1]
            gt_centy = gt[0] + 0.5 * gt_h
            gt_centx = gt[1] + 0.5 * gt_w

            a_h = a[2] - a[0]
            a_w = a[3] - a[1]
            a_centy = a[0] + 0.5 * a_h
            a_centx = a[1] + 0.5 * a_w

            rpn_bboxes[ix] = [(gt_centy - a_centy) / a_h, (gt_centx - a_centx) / a_w,
                              np.log(gt_h / a_h), np.log(gt_w / a_w)]
            rpn_bboxes[ix] /= self.rpn_bbox_std_dev
            ix += 1
        return rpn_match, rpn_bboxes
"""

import numpy as np


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + 0.5 * width
    ctr_y = src_bbox[:, 1] + 0.5 * height

    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height

    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (base_ctr_x - ctr_x) / width
    dy = (base_ctr_y - ctr_y) / height
    dw = np.log(base_width / width)
    dh = np.log(base_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


def loc2bbox(src_bbox, loc):
    if src_bbox.shape[0] == 0:
        return np.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
    ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
    w = np.exp(dw) * src_width[:, np.newaxis]
    h = np.exp(dh) * src_height[:, np.newaxis]

    dst_bbox = np.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox


def bbox_iou(bbox_a, bbox_b):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print(bbox_a, bbox_b)
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """
    生成k个基础anchor框(在faster-rcnn中k等于9)
    :param base_size:
    :param ratios:
    :param anchor_scales:
    :return:
    """
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.
    return anchor_base


def enumerate_shifted_anchor_1(anchor_base, feat_stride, height, width):
    raise Exception('已废弃！推荐使用 enumerate_shifted_anchor')
    # 计算网格中心点
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),
                      shift_x.ravel(),shift_y.ravel(),), axis=1)
    # shift 是一个 (N × 4) 的数组，而实际上(:, 0:2)和(:, 2:4)是相等的，这样做是为了下面计算的方便。

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


def enumerate_shifted_anchor(anchor_base, feature_size, feature_stride):
    """
    :anchor_base: generate by generate_anchor_base.
    :feature_size: type(feature_size) == tuple, len(feature_size) == 2,
                   feature_size[0] == feature_height, feature_size[1] == feature_width
    :feature_stride: type(feature_stride) == tuple, len(feature_stride) == 2,
                   feature_stride[0] == feature_height_stride, feature_stride[1] == feature_width_stride
    """
    # 计算网格中心点
    shift_x = np.arange(0, feature_size[1] * feature_stride[1], feature_stride[1])
    shift_y = np.arange(0, feature_size[0] * feature_stride[0], feature_stride[0])
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_x.ravel(),shift_y.ravel(),
                      shift_x.ravel(),shift_y.ravel(),), axis=1)
    # shift 是一个 (N × 4) 的数组，而实际上(:, 0:2)和(:, 2:4)是相等的，这样做是为了下面计算的方便。

    # 每个网格点上的9个先验框
    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((K, 1, 4))
    # 所有的先验框
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        argmax_ious, label = self._create_label(anchor, bbox)
        # 利用先验框和其对应的真实框进行编码
        loc = bbox2loc(anchor, bbox[argmax_ious])

        return loc, label

    def _create_label(self, anchor, bbox):
        # 1是正样本，0是负样本，-1忽略
        label = np.empty((len(anchor),), dtype=np.int32)
        label.fill(-1)

        # argmax_ious为每个先验框对应的最大的真实框的序号
        # max_ious为每个真实框对应的最大的真实框的iou
        # gt_argmax_ious为每一个真实框对应的最大的先验框的序号
        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox)

        # 如果小于门限函数则设置为负样本
        label[max_ious < self.neg_iou_thresh] = 0

        # 每个真实框至少对应一个先验框
        label[gt_argmax_ious] = 1

        # 如果大于门限函数则设置为正样本
        label[max_ious >= self.pos_iou_thresh] = 1

        # 判断正样本数量是否大于128，如果大于的话则去掉一些
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # 平衡正负样本，保持总数量为256
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox):
        # 计算所有
        ious = bbox_iou(anchor, bbox)
        # 行是先验框，列是真实框
        argmax_ious = ious.argmax(axis=1)
        # 找出每一个先验框对应真实框最大的iou
        max_ious = ious[np.arange(len(anchor)), argmax_ious]
        # 行是先验框，列是真实框
        gt_argmax_ious = ious.argmax(axis=0)
        # 找到每一个真实框对应的先验框最大的iou
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        # 每一个真实框对应的最大的先验框的序号
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious

