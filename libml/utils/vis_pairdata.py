# -*- coding: utf-8 -*-
# @Time    : 2020/4/20
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : vis_datasets.py

import cv2
import numpy as np
import os
import os.path as osp
import sys
from tqdm import tqdm
import json


def display(root, apath, bpath, _w_size=512):
    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', _w_size, _w_size)
    
    abs_a = osp.join(root, apath)
    abs_b = osp.join(root, bpath)
    if osp.isdir(abs_a) and osp.isdir(abs_b):
        imgs_a = [osp.join(abs_a, i) for i in os.listdir(abs_a)]
        imgs_b = [osp.join(abs_b, i) for i in os.listdir(abs_b)]
    elif osp.isfile(abs_a) and osp.isfile(abs_b):
        imgs_a = [abs_a]
        imgs_b = [abs_b]
    else:
        raise  NotImplementedError
    imgs = sorted(imgs_a + imgs_b, key=lambda x: x.split('/')[-1])
    nums = len(imgs)
    index = 0
    while True:
        if index > nums or index < 0:
            index = 0
        imgname = imgs[index]
        img = cv2.imread(imgname)
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.resize(img, (512, 512), fx=0.5, fy=0.5)
        cv2.putText(img, imgname.split('/')[-2], (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k == ord('d'):
            index += 1
        elif k == ord('a'):
            index -= 1
        elif k == ord('s'):
            print(imgname)
        elif k == ord('q'):
            cv2.destroyAllWindows()
            raise KeyError('*** STOP *****')


def display_fromjson(jpath):
    jfile = json.load(open(jpath))
    imgs_a = []
    imgs_b = []
    losses = []
    for sample in jfile:
        imgs_a.append(sample['apath'])
        imgs_b.append(sample['bpath'])
        losses.append(sample['loss'])
        losses.append(sample['loss'])

    cv2.namedWindow('img', 0)
    cv2.resizeWindow('img', 512, 512)
    imgs = sorted(imgs_a + imgs_b, key=lambda x: x.split('/')[-1])
    nums = len(imgs)
    index = 0
    while True:
        if index > nums:
            index = 0
        imgname = imgs[index]
        img = cv2.imread(imgname)
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        img = cv2.resize(img, (512, 512), fx=0.5, fy=0.5)
        cv2.putText(img, imgname.split('/')[-2], (20,20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.putText(img, "%0.4f" % float(losses[index]), (20,40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('img', img)
        k = cv2.waitKey(0)
        if k == ord('d'):
            index += 1
        elif k == ord('a'):
            index -= 1
        elif k == ord('s'):
            print('*' * 20)
            print(imgname)
            print('*' * 20)
        elif k == ord('q'):
            cv2.destroyAllWindows()
            raise KeyError('*** STOP *****')


if __name__ == '__main__qiaonasen':
    # root = '/home/lafe/datasets/cert-link-5000-crop_warp_filtrate'
    # root = "/home/lafe/datasets/a2b_data/cert-link-154197-crop_filtrate"
    # root = "/home/lafe/datasets/a2b_data/cert-link-154197-standard-crop_420"
    #root = '/home/lafe/datasets/a2b_data/cert-3-type/mt_cert-bg2_blue_5-5'
    #root = '/home/lafe/datasets/a2b_data/cert_wholebody_nopad'
    #root = '/home/lafe/datasets/a2b_data/liq'
    root = '/home/lafe/datasets/a2b_data/cert_wholebody_nopad'

    #root = '/home/lafe/datasets/a2b_data/slice_liqued_face_6_17'
    #root = '/home/lafe/datasets/a2b_data/image_and_landmarks_meta'
    #root = '/home/lafe/datasets/a2b_data/cert_wholebody_nopad'
    # root = '/home/lafe/datasets/blue_bg_meta'
    #apath = 'train_A_FFG_nopad_ali'
    #bpath = 'train_B_FFG_nopad_ali'
    #apath = 'meta_original'
    #bpath = 'meta_finish'
    apath = 'train_a2bgrid_A_filt'
    bpath = 'train_a2bgrid_B_filt'
    #apath = 'train_B_FFG_nopad_ali'
    display(root, apath, bpath, _w_size=512)
    imga = '/home/lafe/datasets/a2b_data/cert-3-type/mt_cert-bg2_blue_5-5/train_A_nopad_ali/1003.jpg'
    imgb = '/home/lafe/datasets/a2b_data/cert-3-type/mt_cert-bg2_blue_5-5/train_B_nopad_ali/1003.jpg'
    # display(root, imga, imgb)
    # jpath = '/home/lafe/datasets/a2b_data/cert-3-type/mt_cert-bg2_blue_5-5/mt_cert-bg2_blue_5-5_unmath.json'
    jpath = '/home/lafe/datasets/a2b_data/cert-3-type/mt_cert-bg2_blue_5-5/mt_cert-bg2_blue_5-5_unmath_nopad.json'
    jpath = '/home/lafe/datasets/a2b_data/cert-3-type/mt_cert-bg2_blue_5-5/mt_cert-bg2_blue_5-5_unmath_FFG_nopad.json'
    jpath = '/home/lafe/datasets/a2b_data/cert_wholebody_nopad/cert_wholebody_nopad.json'
    #display_fromjson(jpath)


if __name__ == '__main__':
    root = '/home/dev_sdc/autops_data/liquid_pairs'
    apath = 'A'
    bpath = 'B'
    display(root, apath, bpath, _w_size=512)

