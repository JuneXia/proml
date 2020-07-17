import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys
import time


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






from tqdm import tqdm
if __name__ == '__main__':
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


