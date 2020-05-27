from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from libml.datasets.data_loader import DataLoaderX
from face_detection.retinaface.detector import Detector
import os
import cv2
import numpy as np
import glob
import multiprocessing
import time
from libml.utils import tools

BATCH_SIZE = 1
ALIGN_SIZE = (182, 182)
MARGIN = 44
# DATA_DIR = '/disk1/home/xiaj/res/face/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44'
# DATA_DIR = '/disk1/home/xiaj/res/face/VGGFace2/train'
# DATA_DIR = '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44'
# ALIGN_SAVE_DIR = '/disk1/home/xiaj/res/face/tmp'

DATA_DIR = '/disk1/home/xiaj/res/face/Trillion Pairs/train_msra/msra'
ALIGN_SAVE_DIR = '/disk1/home/xiaj/res/face/Trillion Pairs/train_msra/Experiment/retinaface_align182x182_margin44'
SAVE_FORMAT = '.png'
OVERRIDE = False

NUM_PROCESS = 18  # 一共要启多少个进程
MAX_PROCESS_PER_GPU = 6  # 每个gpu最多能跑几个进程, 3



def process(*args, **kwargs):
    """
    :param process_num: 进程编号
    :param imlist: 待处理的图片列表
    :return:
    """
    process_num, imlist = args
    device = 'cuda:' + str(process_num // MAX_PROCESS_PER_GPU + 1)
    pname = multiprocessing.current_process().name
    print('current_process: {}, device: {}'.format(pname, device))

    fdetector = Detector()

    for i, impath in enumerate(imlist):
        align_path = impath.replace(DATA_DIR, ALIGN_SAVE_DIR)
        if SAVE_FORMAT is not None:
            assert isinstance(SAVE_FORMAT, str) and len(SAVE_FORMAT) >= 4 and SAVE_FORMAT[0] == '.'
            align_path, imext = align_path.rsplit('.', maxsplit=1)
            align_path += SAVE_FORMAT
        if (not OVERRIDE) and os.path.exists(align_path):
            if i % 100 == 0:
                print('align_path: {} exists!'.format(align_path))
            continue

        aligned_bboxes, aligned_images = fdetector.align(impath, align_size=ALIGN_SIZE, margin=MARGIN,
                                                         min_face_area=0, remove_inner_face=False)
        if (aligned_images is None) or (len(aligned_images) == 0):
            continue
        assert len(aligned_images) == 1, "目前每张图片只取中间的一张人脸"

        aligned_images = aligned_images[0].astype(np.uint8)

        cv2.imwrite(align_path, aligned_images)

        # save multi face
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # for j in range(len(aligned_images)):
        #     pass
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # cv2.imshow('show', aligned_images)
        # cv2.waitKey()
        if i % 100 == 0:
            pname = multiprocessing.current_process().name
            tools.view_bar('{}: '.format(pname), i, len(imlist))


if __name__ == '__main__':
    print("\nThe number of CPU is:" + str(multiprocessing.cpu_count()))

    print('\nStep1: glob load data from {}'.format(DATA_DIR))
    images_path = glob.glob(os.path.join(DATA_DIR, '*/*'))

    print('\nStep2> Make align dir:')
    for i, impath in enumerate(images_path):
        align_path = impath.replace(DATA_DIR, ALIGN_SAVE_DIR)
        align_dir, imname = align_path.rsplit('/', maxsplit=1)
        if not os.path.exists(align_dir):
            os.makedirs(align_dir)

        tools.view_bar('Make align dir: ', i, len(images_path))
    print('')

    print('\nStep3> Create multiprocess:')
    batch_size = len(images_path) // NUM_PROCESS
    for i in range(NUM_PROCESS):
        if i+1 == NUM_PROCESS:
            end_index = None
        else:
            end_index = (i + 1) * batch_size
        imlist = images_path[i*batch_size:end_index]
        proc = multiprocessing.Process(target=process, args=(i, imlist))
        proc.start()
        time.sleep(40)

    print('\nStep4> Main Process join:')
    for p in multiprocessing.active_children():
        p.join()
        print("child   p.name: {} \t pid: {} \t finish".format(p.name, str(p.pid)))
    print("END!!!!!!!!!!!!!!!!!")

