import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print('POJECT_ROOT: ', PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

import shutil
import argparse
import cv2
from face_detection import FaceDetector
from utils import tools


def face_filter(src_path, dst_path):
    people_list = os.listdir(src_path)
    for i, people in enumerate(people_list):
        people_image_path = os.path.join(src_path, people)
        people_image_list = os.listdir(people_image_path)
        for image_name in people_image_list:
            image_path = os.path.join(people_image_path, image_name)
            print(image_path)
            bboxes = fdetector.detect(image_path, remove_inner_face=False)
            if len(bboxes) == 0:
                if dst_path is None:
                    os.remove(image_path)
                else:
                    people_image_dstpath = os.path.join(dst_path, people)
                    if not os.path.exists(people_image_dstpath):
                        os.makedirs(people_image_dstpath)
                    dst_image_path = os.path.join(people_image_dstpath, image_name)
                    shutil.move(image_path, dst_image_path)
            elif len(bboxes) > 1:
                img = cv2.imread(image_path)
                for j, bbox in enumerate(bboxes):
                    x1, y1, x2, y2 = bbox
                    width_delta = (x2 - x1)
                    height_delta = (y2 - y1)
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))

                    x1 -= width_delta
                    y1 -= height_delta
                    x2 += width_delta
                    y2 += height_delta
                    x1 = max(int(x1), 0)
                    y1 = max(int(y1), 0)
                    x2 = min(int(x2), img.shape[1])
                    y2 = min(int(y2), img.shape[0])
                    # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0))
                    imsave = img[y1:y2, x1:x2, :]
                    crop_image_name = image_name.split('.')[0] + '-' + str(j) + '.jpg'
                    image_path = os.path.join(people_image_path, crop_image_name)
                    cv2.imwrite(image_path, imsave)
                # cv2.imshow('show', img)
                # cv2.waitKey(0)

        tools.view_bar('face_filter: ', i + 1, len(people_list))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--src_path', type=str, help='src people image path')
    parser.add_argument('--dst_path', type=str, help='move to dst people image path. remove image if this param is none.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    sys.argv = ['face_filter.py',
                '--src_path', '/home/xiajun/res/face/GC-WebFace/raw',
                '--dst_path', '/home/xiajun/res/face/GC-WebFace-tmp',
                ]

    args = parse_arguments(sys.argv[1:])
    print(args)
    src_path = args.src_path
    dst_path = args.dst_path

    fdetector = FaceDetector()

    face_filter(src_path, dst_path)

print('debug')