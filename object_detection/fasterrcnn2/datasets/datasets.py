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
import os
from PIL import Image
import keras.engine as KE
from torch.utils.data import Dataset
import torch

random.seed(1)


class ShapeData():
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


class VOC2007Dataset(Dataset):
    def __init__(self, vocfile, transforms):

        self.vocfile = vocfile
        self.transforms = transforms
        # self.img_dir = os.path.join(data_dir, "PNGImages")
        # self.txt_dir = os.path.join(data_dir, "Annotation")
        # self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".png"), os.listdir(self.img_dir)))]

        self.images_path, self.images_label = self.load_data(self.vocfile)

    def load_data(self, vocfile):
        with open(vocfile, 'r') as f:
            lines = f.readlines()

        images_path = list()
        images_label = list()
        for line in lines:
            image_info = line.strip().split(' ')
            images_path.append(image_info[0])
            image_label = [iminfo.split(',') for iminfo in image_info[1:]]
            image_label = np.array(image_label).astype(np.int32)
            images_label.append(image_label)

        return images_path, images_label

    def __getitem__(self, index):
        """
        返回img和target
        :param idx:
        :return:
        """

        # name = self.names[index]
        # path_img = os.path.join(self.img_dir, name + ".png")
        # path_txt = os.path.join(self.txt_dir, name + ".txt")
        image_path = self.images_path[index]
        image_bbox = self.images_label[index][:, 0:4]
        image_label = self.images_label[index][:, -1]
        img = Image.open(image_path).convert("RGB")

        # load boxes and label
        # f = open(path_txt, "r")
        # import re
        # points = [re.findall(r"\d+", line) for line in f.readlines() if "Xmin" in line]
        # boxes_list = list()
        # for point in points:
        #     box = [int(p) for p in point]
        #     boxes_list.append(box[-4:])

        boxes = torch.tensor(image_bbox, dtype=torch.float)
        labels = torch.tensor(image_label, dtype=torch.long)

        # iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if len(self.images_path) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(self.vocfile))
        return len(self.images_path)
