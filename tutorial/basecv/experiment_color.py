import os
import cv2


def scan_file(path):
    files = []
    for file in os.listdir(path):
        abs_path = os.path.join(path, file)
        if not os.path.isfile(abs_path):
            continue
        files.append(abs_path)
    return files


if __name__ == '__main__':
    data_path = '/home/dev_sdc/autops_data/liquid_pairs-with_lm'
    imgs_path = scan_file(data_path)

    for impath in imgs_path:
        pass


    ref_width = 512
    img = cv2.imread(impath)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_w > ref_width:
        scale = ref_width / img_w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue = img_hsv[50:100, 50:100, 0].mean()

    pt1 = (50, 50)
    pt2 = (100, 100)
    img = cv2.rectangle(img, pt1, pt2, (0, 0, 255), 2)
    img_hsv = cv2.rectangle(img_hsv, pt1, pt2, (0, 0, 255), 2)

    cv2.imshow('bgrimg', img)
    cv2.imshow('hsvimg', img_hsv)
    cv2.waitKey()

    print('debug')