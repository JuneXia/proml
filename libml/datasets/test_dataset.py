# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from matplotlib import pyplot as plt
import cv2
import time
from libml.datasets.dataset import Dataset1 as Dataset
from libml.datasets.data_loader import DataPrefetcher as data_prefetcher
from tutorial.tools.common_tools import transform_invert
from libml.datasets.data_loader import DataLoaderX

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
input_size = 300
margin_scale = 182/160
# DATA_DIR = '/disk2/res/face/Trillion Pairs/train_msra/msra'
DATA_DIR = '/disk1/home/xiaj/res/face/VGGFace2/Experiment/mtcnn_align182x182_margin44'
# DATA_DIR = '/disk2/res/face/CASIA-FaceV5/CASIA-FaceV5-000-499-mtcnn_align182x182_margin44'
BATCH_SIZE = 128

train_transform = transforms.Compose([
    # transforms.Resize((182, 182)),
    # transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale))),
    # transforms.RandomApply([transforms.Resize((30, 30)), transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale)))], p=0.1),
    # transforms.RandomCrop(160),
    # transforms.RandomCrop(input_size),
    # transforms.RandomRotation((10), expand=True),

    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(contrast=0.5),
    # transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    # transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), shear=(-10, 10, -10, 10)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 45, 0, 0)),

    # transforms.RandomGrayscale(p=0.1),

    transforms.ToTensor(),
    # transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),

    # transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

local_rank = 0
cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(local_rank)
torch.set_printoptions(precision=10)

torch.cuda.set_device(0)

os.environ.setdefault("MASTER_ADDR", "10.10.2.199")
os.environ.setdefault("MASTER_PORT", "10022")
torch.distributed.init_process_group(backend='nccl', world_size=4, rank=local_rank)
world_size = torch.distributed.get_world_size()

# train_data = Dataset(data_dir=DATA_DIR, transform=train_transform)
train_data = datasets.ImageFolder(
    DATA_DIR,
    transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), # Too slow
            # normalize,
        ]))
train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

# train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
train_loader = DataLoaderX(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=1, pin_memory=True)

# prefetcher = data_prefetcher(train_loader)

if __name__ == '__main__':
    time_dict = dict()
    t1 = time.time()
    for i, data in enumerate(train_loader):
        t2 = time.time()
        print('DataLoad: ', t2 - t1)
        inputs, labels = data
        time.sleep(0.4)

        # 返回单个图片的可视化方法
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # img_tensor = inputs[0, ...]  # C H W
        # img = transform_invert(img_tensor, train_transform)
        # img = np.array(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imshow('show', img)
        # cv2.waitKey()
        t1 = time.time()
        continue
        plt.imshow(img)
        plt.show()
        plt.pause(0.5)
        plt.close()
        continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == '__main__':
    time_dict = dict()

    data = prefetcher.next()
    while data is not None:
        # inputs, labels = data
        time.sleep(0.2)
        t1 = time.time()
        data = prefetcher.next()
        t2 = time.time()
        print('DataLoad: ', t2 - t1)