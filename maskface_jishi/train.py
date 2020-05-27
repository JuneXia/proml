import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset_akou as dataset
import transforms
from tutorial.tools import common_tools
from matplotlib import pyplot as plt
from libml.models.mobilenet import MobileNetV1
from retinaface import RetinaFace
from losses import MultiBoxLoss
from prior_box import PriorBox
from config import cfg_mnet as cfg
import numpy as np
import cv2
import time

IMG_DIM = cfg['image_size']
# RGB_MEAN = (104, 117, 123) # bgr order
RGB_MEAN = (0, 0, 0)  # bgr order

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

BATCH_SIZE = 64
MAX_EPOCH = 100
num_classes = cfg['num_classes']
LR = 0.01
log_interval = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    images, targets = tuple(zip(*batch))
    # targets_ = tuple()
    # for t in targets:
    #     target = dict()
    #     for k, v in t.items():
    #         if v is None:
    #             target[k] = v
    #         else:
    #             target[k] = torch.tensor(v)
    #     targets_ += (target, )
    return torch.stack(images, 0), targets


if __name__ == '__main__':
    data_dir = '/disk1/home/xiaj/res/face/maskface/zhihu_akou'

    # train_transform = transforms.Compose(IMG_DIM, RGB_MEAN)
    train_transform = transforms.Compose1([
        transforms.Crop(IMG_DIM),
        transforms.Distort(),
        transforms.Pad2Square(fill=RGB_MEAN),  # TODO: 以后这个RGB_MEAN需要删除，减均值操作直接交给 transforms.Normalize
        transforms.RandomMirror(p=0.5),
        transforms.Resize(min_size=IMG_DIM, max_size=IMG_DIM),
        transforms.ToTensor(),
    ])
    train_dataset = dataset.MaskFaceAkouDataset(data_dir, train_transform)

    epoch_size = math.ceil(len(train_dataset) / BATCH_SIZE)
    max_iter = MAX_EPOCH * epoch_size

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=6)

    # net = MobileNetV1(num_classes=num_classes)
    net = RetinaFace(cfg=cfg).to(device)
    params = [p for p in net.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(IMG_DIM, IMG_DIM))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    for epoch in range(MAX_EPOCH):
        lr = lr_scheduler.get_lr()[-1]
        t1 = time.time()
        time_dict = dict()
        for i, data in enumerate(train_loader):
            t2 = time.time()
            time_dict['iterdat'] = t2 - t1
            print(time_dict['iterdat'])
            continue

            images, targets = data  # B C H W
            t1 = time.time()
            images = images.to(device)
            t2 = time.time()
            time_dict['imcu'] = t2 - t1

            t1 = time.time()
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets_ = list()
            for t in targets:
                target = dict()
                for k, v in t.items():
                    if v is None:
                        target[k] = v
                    else:
                        target[k] = v.to(device)
                targets_ += (target, )
            targets = targets_
            t2 = time.time()
            time_dict['TargetCu'] = t2 - t1

            t1 = time.time()
            outputs = net(images)
            t2 = time.time()
            time_dict['NetFW'] = t2 - t1

            # img = images[0]  # C H W
            # img = np.array(img)
            # # img = np.transpose(img, [1, 2, 0]).astype(np.uint8)
            #
            # boxes = targets[0]['boxes']
            # for i in range(len(boxes)):
            #     box = boxes[i].astype(np.int)
            #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 255, 0))
            # labels = targets[0]['labels']
            # text = '{} labels'.format(len(labels))
            # cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255))
            #
            # cv2.imshow('show', img)
            # cv2.waitKey()

            # backward

            optimizer.zero_grad()

            t1 = time.time()
            loss_l, loss_c = criterion(outputs, priors, targets)
            t2 = time.time()
            time_dict['LossFW'] = t2 - t1

            loss = cfg['loc_weight'] * loss_l + loss_c

            t1 = time.time()
            loss.backward()
            t2 = time.time()
            time_dict['LossBKW'] = t2 - t1

            # update weights
            t1 = time.time()
            optimizer.step()
            t2 = time.time()
            time_dict['UpdateGrad'] = t2 - t1


            # plt.imshow(img)
            # plt.show()
            # plt.pause(0.5)
            # plt.close()

            # images = list(image.to(device) for image in images)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if (i + 1) % log_interval == 0:
                text = "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Lloc: {:.4f} Lcls:{:.4}".format(
                    epoch, MAX_EPOCH, i + 1, len(train_loader), loss_l, loss_c)
                text += " Lr:{:.4f}".format(lr)

                print(text)

            for k, v in time_dict.items():
                print('{}:{:.3f}'.format(k, v), end=' | ')
            print('')
            time_dict = dict()
            t1 = time.time()
