import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
import time
from tqdm import tqdm


def resize(img_path):
    img = cv2.imread(img_path)
    scale = 256 / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img


def cut(img_path):
    img = cv2.imread(img_path)
    scale = 256 / img.shape[1]
    img = cv2.resize(img, None, fx=scale, fy=scale)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)
    t1 = time.time()
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 20, cv2.GC_INIT_WITH_RECT)
    print('grabCut time: ', time.time() - t1)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img), plt.colorbar(), plt.show()


def cut_water(img_path):
    # img = cv2.imread(img_path)
    img = resize(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # plt.imshow(sure_bg), plt.colorbar(), plt.show()


    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # plt.imshow(sure_fg), plt.colorbar(), plt.show()


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    # plt.imshow(markers), plt.colorbar(), plt.show()

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    plt.imshow(markers), plt.colorbar(), plt.show()


if __name__ == '__main__1':
    sys.argv = ['main.py',
                'test_img/4ea0ccd2c679f00fe28f44f4abababd9_original_photo.jpg'
                ]
    print('sys.argv: ', sys.argv)
    args = sys.argv[1:]
    img_path = args[0]

    # 阿里云部署
    # root_imgs = '/home/qiaonasen/tangni/crop_service/test/test_img1'
    # save_path = '/home/qiaonasen/tangni/tmp'

    # mainto-250
    # root_imgs = '/home/mtbase/tangni/res/test_img4autops_crop'
    # save_path = '/home/mtbase/tangni/tmp'

    root_imgs = '/home/dev_sdb/temp11'
    save_path = '/home/dev_sdb/autops_crop_result'

    imgs_path = [os.path.join(root_imgs, i) for i in os.listdir(root_imgs)]
    # imgs=glob.glob('test_imgs/1-3.jpg')
    # imgs=imgs[0:1]
    t = time.time()
    for impath in tqdm(imgs_path):
        if impath in ['/home/dev_sdb/temp11/1f459a6130716b5eb71d9c181368ee63_original_photo.jpg']:
            print('debug')
            continue
        # impath = '/home/dev_sdb/temp11/c348fe3bc9e63ba17d2d61de7f9c175d_original_photo.jpg'
        # impath = '/home/dev_sdb/temp11/02497ccef7822c63f721a4d0e7653fc2_original_photo.jpg'
        # impath = '/home/dev_sdb/temp11/Z64e3e37d3444fb5805c4d7ddfc96ca2b_original_photo.jpg'
        # img_path = '/home/mtbase/tangni/czb_algo_service_mainto/pt_autops_crop_service/test/64e3e37d3444fb5805c4d7ddfc96ca2b_original_photo.jpg'
        imsavepath = os.path.join(save_path, impath.rsplit('/', maxsplit=1)[-1])
        if os.path.exists(imsavepath):
            print("img path exited: {}".format(imsavepath))
            # continue

        print("img path:{}".format(impath))

        cut_water(impath)

    print("cost:{}".format(time.time() - t))


if __name__ == '__main__x':  # 形态学运算实验
    impath = '/home/mtbase/tangni/res/himo_credential_blue.jpg'

    ref_width = 512
    img = cv2.imread(impath)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_w > ref_width:
        scale = ref_width / img_w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # img = cv2.Canny(img, 150, 100, 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (9, 9))
    _, bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    canny = cv2.Canny(blur, 3, 9, 3)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph_open = cv2.morphologyEx(canny, cv2.MORPH_OPEN, element)
    morph_close = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, element)
    morph_grad = cv2.morphologyEx(canny, cv2.MORPH_GRADIENT, element)
    morph_tophat = cv2.morphologyEx(canny, cv2.MORPH_TOPHAT, element)
    morph_blackhat = cv2.morphologyEx(canny, cv2.MORPH_BLACKHAT, element)
    morph_erode = cv2.morphologyEx(canny, cv2.MORPH_ERODE, element)
    morph_dilate = cv2.morphologyEx(canny, cv2.MORPH_DILATE, element)

    if True:
        cv2.imshow('blur', blur)
        cv2.imshow('bin', bin)
        cv2.imshow('canny', canny)
        cv2.imshow('morph_open', morph_open)
        cv2.imshow('morph_close', morph_close)
        cv2.imshow('morph_grad', morph_grad)
        cv2.imshow('morph_tophat', morph_tophat)
        cv2.imshow('morph_blackhat', morph_blackhat)
        cv2.imshow('morph_erode', morph_erode)
        cv2.imshow('morph_dilate', morph_dilate)

        cv2.waitKey()


DEBUG = True


def draw_contours(contours, background):
    for cont in contours:
        # find bounding box coordinates
        img_tmp = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(img_tmp, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # find minimum area
        rect = cv2.minAreaRect(cont)
        # calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)
        # normalize coordinates to integers
        box = np.int0(box)
        # draw contours
        cv2.drawContours(img_tmp, [box], 0, (0, 0, 255), 3)

        # calculate center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cont)
        # cast to integers
        center = (int(x), int(y))
        radius = int(radius)
        # draw the circle
        img = cv2.circle(img_tmp, center, radius, (0, 255, 0), 2)

        cv2.imshow('show', img_tmp)
        cv2.waitKey()

    # image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None
    cv2.drawContours(background, contours, -1, (255, 0, 0), thickness=-1)
    cv2.imshow("contours", background)

    # cv2.waitKey()
    # cv2.destroyAllWindows()

def get_background(img):
    blur = cv2.medianBlur(img, 7)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    edge = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
    norm_invers_alpha = (1.0 / 255) * (255 - gray)
    channels = cv2.split(img)
    for ch in channels:
        ch[:] = ch * norm_invers_alpha
    merger = cv2.merge(channels)

    # blur = cv2.medianBlur(merger, 7)
    # gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    # edge = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)

    ret, edge = cv2.threshold(edge, 0, 255, cv2.THRESH_OTSU)

    if DEBUG:
        # cv2.imshow('img', img)
        # cv2.imshow('gray', gray)
        # cv2.imshow('edge', edge)
        cv2.imshow('merger', merger)
        # cv2.waitKey()
    #
    # continue

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    morph = cv2.morphologyEx(edge, cv2.MORPH_DILATE, element)
    # ret, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # contourscontours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    foreground = np.zeros_like(edge)
    cv2.drawContours(foreground, contours, -1, (255,), thickness=-1)
    background = 255 - foreground

    img_fore = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=foreground)
    img_back = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=background)

    areas = [cv2.contourArea(cont) for cont in contours]
    if len(contours) > 1:
        print('debug')

    contour = contours[np.argsort(areas)[-1]]

    x, y, w, h = cv2.boundingRect(contour)

    patch_w = patch_h = 32
    if x <= patch_w:
        px, py = 100, 100
    else:
        px, py = int(x/2 - patch_w/2), y
        # px, py = int(x-20 - patch_w/2), y

    patch_rect = (px, py, patch_w, patch_h)

    if True:
        template = background.copy()
        cv2.imshow('src_template', template)
        cv2.waitKey()

        # fore_patch_tmp = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(fore_patch_tmp, (patch_rect[0], patch_rect[1]), (patch_rect[0]+patch_rect[2], patch_rect[1]+patch_rect[3]), (0, 255, 0), 2)
        # cv2.imshow('fore_patch', fore_patch_tmp)
        # cv2.waitKey()

        template[0:py, :] = 0
        template[py + patch_h:, :] = 0
        template[:, :px] = 0
        template[:, px + patch_w:] = 0

        cv2.imshow('dst_template', template)
        cv2.waitKey()

    patch = img[py:py+patch_h, px:px+patch_w, :]
    mask = background[py:py+patch_h, px:px+patch_w]
    patch_back = cv2.add(patch, np.zeros(np.shape(patch), dtype=np.uint8), mask=mask)

    if True:
        cv2.imshow('mask', mask)
        cv2.imshow('patch', patch)
        cv2.imshow('patch_back', patch_back)
        cv2.waitKey()

    # background = np.pad(background, pad_width=100, mode='constant')
    # contours, hierarchy = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if False:
        draw_contours([contour], background)


    if False:
        cv2.imshow('img', img)
        cv2.imshow('gray', gray)
        cv2.imshow('edge', edge)
        cv2.imshow('morph', morph)
        cv2.imshow('foreground', foreground)
        cv2.imshow('background', background)
        cv2.imshow('img_fore', img_fore)
        cv2.imshow('img_back', img_back)

        cv2.waitKey()

    return patch_back


if __name__ == '__main__':  # 通过一些形态学算法提取前景区域
    data_path = '/home/dev_sdc/resources/职业照底色/芽黄'  # 深灰
    # data_path = '/home/dev_sdc/autops_data/liquid_pairs-with_lm/B'
    ref_width = 5120
    for imname in os.listdir(data_path):
        impath = os.path.join(data_path, imname)
        # impath = '/home/dev_sdc/resources/职业照底色/芽黄/ll2umq1Witac_tSeBH1mnSNB5v8o.jpg'
        print(impath)

        img = cv2.imread(impath)
        img_h, img_w = img.shape[0], img.shape[1]
        img = img[0:img_h//3, :, :]
        img_h, img_w = img.shape[0], img.shape[1]
        if img_w > ref_width:
            scale = ref_width / img_w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # img = cv2.Canny(img, 150, 100, 3)

        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # blur = cv2.blur(gray, (5, 5))
        # blur = gray
        # # edge = cv2.Canny(blur, 3, 9, 3)
        # grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)
        # grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)
        # absX = cv2.convertScaleAbs(grad_x)  # 转回uint8
        # absY = cv2.convertScaleAbs(grad_y)
        #
        # edge = cv2.addWeighted(absX, 0.5, absY, 0.5, 2)
        # ret, edge = cv2.threshold(edge, 0, 255, cv2.THRESH_OTSU)

        get_background(img)

def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect

if __name__ == '__main__x':
    data_path = '/home/dev_sdc/resources/职业照底色/芽黄'
    data_path = '/home/dev_sdc/autops_data/liquid_pairs-with_lm/B'
    ref_width = 512
    for imname in os.listdir(data_path):
        impath = os.path.join(data_path, imname)

        img = cv2.imread(impath)
        img_h, img_w = img.shape[0], img.shape[1]
        img = img[0:img_h//2, :, :]
        img_h, img_w = img.shape[0], img.shape[1]
        if img_w > ref_width:
            scale = ref_width / img_w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # img = cv2.Canny(img, 150, 100, 3)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (9, 9))
        canny = cv2.Canny(blur, 3, 9, 3)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        morph = cv2.morphologyEx(canny, cv2.MORPH_DILATE, element)
        # ret, bin = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # contourscontours, hierarchy = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        foreground = np.zeros_like(canny)
        cv2.drawContours(foreground, contours, -1, (255, ), thickness=-1)
        background = 255 - foreground

        background = np.pad(background, pad_width=100, mode='constant')
        contours, hierarchy = cv2.findContours(background, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        cv2.imshow('foreground', foreground)
        cv2.imshow('background', background)
        cv2.waitKey()

        # c = sorted(contours, key=cv2.contourArea, reverse=True)[0]  # 先找出轮廓点
        #
        # rect = order_points(c.reshape(c.shape[0], 2))
        # rect = rect.astype(dtype=np.int)
        #
        # xs = [i[0] for i in rect]
        # ys = [i[1] for i in rect]
        # xs.sort()
        # ys.sort()
        # # 内接矩形的坐标为
        # print(xs[1], xs[2], ys[1], ys[2])
        #
        # img_tmp = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        # cv2.rectangle(img_tmp, (xs[1], ys[1]), (xs[2], ys[2]), (0, 255, 0), 2)

        for cont in contours:
            img_tmp = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
            for i, c in enumerate(cont):
                c = c.reshape((2,)).tolist()
                img = cv2.circle(img_tmp, (c[0], c[1]), 2, (0, 255, 0), 2)
                # cv2.imshow('show', img)
                # cv2.waitKey(10)

            cv2.imshow('show2', img)
            cv2.waitKey()






            # find bounding box coordinates
            img_tmp = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
            x, y, w, h = cv2.boundingRect(cont)
            cv2.rectangle(img_tmp, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # find minimum area
            rect = cv2.minAreaRect(cont)
            # calculate coordinates of the minimum area rectangle
            box = cv2.boxPoints(rect)
            # normalize coordinates to integers
            box = np.int0(box)
            # draw contours
            cv2.drawContours(img_tmp, [box], 0, (0, 0, 255), 3)

            # calculate center and radius of minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(cont)
            # cast to integers
            center = (int(x), int(y))
            radius = int(radius)
            # draw the circle
            img = cv2.circle(img_tmp, center, radius, (0, 255, 0), 2)

            cv2.imshow('show', img_tmp)
            cv2.waitKey()

        # image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None
        cv2.drawContours(background, contours, -1, (255, 0, 0), thickness=-1)
        cv2.imshow("contours", background)

        # cv2.waitKey()
        # cv2.destroyAllWindows()


        if True:
            cv2.imshow('blur', blur)
            cv2.imshow('morph', morph)
            cv2.imshow('canny', canny)
            cv2.imshow('foreground', foreground)
            cv2.waitKey()

