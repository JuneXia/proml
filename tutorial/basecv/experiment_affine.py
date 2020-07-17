import cv2
import numpy as np


if __name__ == '__main__1':  # 图像缩放，坐标缩放
    srcimg = cv2.imread("/home/mtbase/tangni/res/lena.png", 0)  # 读取灰度图
    h, w = srcimg.shape
    resize_ratio = (0.5, 0.5)  # 定义缩放比例(ratio_h, ratio_w)
    nh, nw = int(h*resize_ratio[0]), int(w*resize_ratio[1])  # 缩放后的宽高
    M = np.mat([[resize_ratio[0], 0], [0, resize_ratio[1]]])  # 缩放矩阵
    resize_img = np.zeros((nw, nh))  # 用于存储缩放后的图像
    points = np.array([(100, 150), (200, 250)], dtype=np.float64)  # 位于原图上的点

    # 坐标点缩放
    new_points = np.dot(points, M)  # 位于缩放后的图像上的点

    # 图像缩放
    for r in range(nh):
        for l in range(nw):
            v = np.dot(M.I, np.array([r, l]).T)
            resize_img[r, l] = srcimg[int(v[0, 0]), int(v[0, 1])]

    # 图像缩放也可以用 opencv 的借口 resize 来做。
    # resize_img = cv2.resize(srcimg, (nw, nh))

    for p in points.astype(np.int32):
        cv2.circle(srcimg, (p[0], p[1]), 2, (255, 0, 0), 1)

    for p in new_points.astype(np.int32):
        cv2.circle(resize_img, (p[0], p[1]), 2, (255, 0, 0), 1)

    cv2.imshow("srcimg", srcimg)
    cv2.imshow("resize_img", resize_img.astype("uint8"))
    cv2.waitKey()


if __name__ == '__main__2':  # 图像、坐标旋转
    img = np.ones((512, 800, 3), dtype=np.uint8) * 128
    img_h, img_w, _ = img.shape
    angle = 10
    cw, ch = img_w // 2, img_h // 2
    rect_size = (360, 200)
    points = np.array([(cw - rect_size[0]/2, ch - rect_size[1]/2), (cw + rect_size[0]/2, ch + rect_size[1]/2)])
    p = points.astype(np.int)
    cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 0, 255))
    cv2.imshow('src_img', img)
    cv2.waitKey()

    M = cv2.getRotationMatrix2D((cw, ch), angle, 1.0)

    # 对图像的旋转
    # img = cv2.warpAffine(img, M, (img_w, img_h), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)

    # 对坐标的旋转
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    points = M.dot(points_ones.T).T
    p = points.astype(np.int)

    cv2.line(img, tuple(p[0]), tuple(p[1]), (0, 255, 0))
    cv2.imshow('rotate_img', img)
    cv2.waitKey()


def xxx(angle):
    tana = math.tan(angle * math.pi / 180.0)
    sina = math.sin(angle * math.pi / 180.0)
    A = 1.0 / tana + 1.0 / sina  # TODO:

    P = (rect_h - A * rect_w) / (tana - A)
    Q = A / (tana - A)

    B = 1 + tana ** 2
    delta = (2 * P * Q * B) ** 2 - 4 * (B - 1) * B * (P ** 2)

    # 得到两个解：
    # c1 = (-2 * P * Q * B + np.sqrt(delta)) / (2 * (B - 1))
    c = (-2 * P * Q * B - np.sqrt(delta)) / (2 * (B - 1))

    a = P + Q * c
    b = a * tana
    d = rect_w - a - c
    print(a, b, c, d)

    return b, d

import math
if __name__ == '__main__3':  # 图像旋转后会有黑边，本实验探索在旋转后的图片中裁剪出不包含黑边的最大区域。
    img = np.zeros((512, 800, 3), dtype=np.uint8)
    img_h, img_w, _ = img.shape
    angle = 10

    cw, ch = img_w // 2, img_h // 2
    rect_w, rect_h = 360, 200
    rect_size = (rect_w, rect_h)
    points = np.array(
        [(cw - rect_size[0] / 2, ch - rect_size[1] / 2), (cw + rect_size[0] / 2, ch + rect_size[1] / 2)])
    p = points.astype(np.int)
    cv2.rectangle(img, tuple(p[0]), tuple(p[1]), (0, 0, 255))

    while True:
        for angle in [10, 20, 30, 40, 50, 60, 70, -10, -20, -30, -40, -50, -60, -70]:
            tana = math.tan(angle * math.pi / 180.0)
            sina = math.sin(angle * math.pi / 180.0)
            A = 1.0/tana + 1.0/sina  # TODO:

            P = (rect_h - A * rect_w) / (tana - A)
            Q = A / (tana - A)

            B = 1 + tana ** 2
            delta = (2*P*Q*B)**2 - 4*(B-1)*B*(P**2)

            c1 = (-2*P*Q*B + np.sqrt(delta))/(2*(B-1))
            c = (-2*P*Q*B - np.sqrt(delta))/(2*(B-1))

            a = P + Q*c
            b = a * tana
            d = rect_w - a - c
            print(a, b, c, d)

            M = cv2.getRotationMatrix2D((cw, ch), angle, 1.0)
            rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            imag = img + rotated_img

            cv2.imshow('rotate_img', imag)
            cv2.waitKey()


if __name__ == '__main__':  # 使用较小的图像矩阵，测试图像旋转后的数值变化
    img = np.array([[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35],
                    [40, 41, 42, 43, 44, 45], [50, 51, 52, 53, 54, 55]], dtype=np.float64)
    img = np.array([img, img, img])
    img = np.reshape(img, (6, 6, 3))
    ang = 1  # 旋转角度
    nw, nh, _ = img.shape
    cx, cy = nw // 2, nh // 2
    m = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)

    if False:  # 参考文献[1]
        cos = np.abs(m[0, 0])
        sin = np.abs(m[0, 1])
        # compute the new bounding dimensions of the image
        nw = int((nh * sin) + (nw * cos))
        nh = int((nh * cos) + (nw * sin))
        # adjust the rotation matrix to take into account translation
        m[0, 2] += (nw / 2) - cx
        m[1, 2] += (nh / 2) - cy

    print('srcimg: \n', img[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT)
    print('default interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
    print('linear interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    print('nearest interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_CUBIC)
    print('cubic interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_AREA)
    print('area interp: \n', wimg[:, :, 0])

    wimg = cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LANCZOS4)
    print('lanczos4 interp: \n', wimg[:, :, 0])

    cv2.imshow('show', wimg)
    cv2.waitKey()