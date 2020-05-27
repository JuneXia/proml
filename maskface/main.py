# -*- coding:utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from matplotlib import pyplot as plt
# from model.lenet import LeNet
from tutorial.model.lenet import LeNet
from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import resnet34
from torchvision.models.densenet import densenet121
from torchvision.models.inception import inception_v3  # failed
from torchvision.models.googlenet import googlenet  # failed
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.shufflenetv2 import shufflenet_v2_x1_5
from torchvision.models.squeezenet import squeezenet1_0
from torchvision.models.squeezenet import squeezenet1_1
from torchvision.models.mnasnet import mnasnet1_0
from torchvision.models.mnasnet import mnasnet0_5
from libml.models.mobilenet import MobileNetV1

import torchvision.models._utils as _utils
from tutorial.tools.common_tools import transform_invert
import dataset
from config import cfg_net
# from tools.common_tools import set_seed
from libml.utils.config import SysConfig
from libml.utils import tools
import cv2


# set_seed()  # 设置随机种子

# 参数设置
MAX_EPOCH = cfg_net['max_epoch']
BATCH_SIZE = cfg_net['batch_size']  # 64 128试试
LR = 0.01
DEVICE = cfg_net['device']
BACKBONE = cfg_net['backbone']
MODEL_SAVE_PATH = 'save_model'
SAVE_FIELD = tools.get_strtime()
log_interval = 30
val_interval = 1
test_interval = 1


# ============================ step 1/5 数据 ============================
csv_file = '/disk1/home/xiaj/res/face/maskface/maskface-training.csv'
test_dir = '/disk1/home/xiaj/res/face/maskface/Experiment/MAFA-test-images-mtcnn_align182x182_margin44/test-images-detected_face-classified'
image_list, label_list, val_image_list, val_label_list = tools.load_data_from_csv(csv_file, all4train=False)


norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
input_size = 96
margin_scale = 182/160

train_transform = transforms.Compose([
    # transforms.Resize((182, 182)),
    transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale))),
    transforms.RandomApply([transforms.Resize((30, 30)), transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale)))], p=0.1),
    # transforms.RandomCrop(160),
    transforms.RandomCrop(input_size),
    # transforms.RandomRotation((10), expand=True),

    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ColorJitter(saturation=0.5),
    # transforms.ColorJitter(hue=0.3),

    transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), shear=(-10, 10, -10, 10)),
    # transforms.RandomAffine(degrees=0, scale=(0.7, 0.7)),
    # transforms.RandomAffine(degrees=0, shear=(0, 45, 0, 0)),

    transforms.RandomGrayscale(p=0.1),

    transforms.ToTensor(),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),

    # transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale))),
    transforms.Resize((input_size, input_size)),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

train_data = dataset.MaskFaceDatasetFormat(format_data=(image_list, label_list), transform=train_transform)
valid_data = dataset.MaskFaceDatasetFormat(format_data=(val_image_list, val_label_list), transform=valid_transform)
test_data = dataset.MaskFaceDataset(test_dir, transform=valid_transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

# ============================ step 2/5 模型 ============================
device = torch.device(DEVICE)  # cpu, cuda:0

# net = LeNet(classes=2)
# net.initialize_weights()

if BACKBONE == 'resnet18':
    net = resnet18(num_classes=2).to(device)
elif BACKBONE == 'resnet34':
    net = resnet34(num_classes=2).to(device)
elif BACKBONE == 'resnet50':
    net = resnet50(num_classes=2).to(device)
elif BACKBONE == 'densenet121':
    net = densenet121(num_classes=2).to(device)
elif BACKBONE == 'mobilenet_v2':
    net = mobilenet_v2(num_classes=2, width_mult=0.35, inverted_residual_setting=None, round_nearest=8).to(device)
elif BACKBONE == 'shufflenet_v2_x1_5':
    net = shufflenet_v2_x1_5(num_classes=2).to(device)
elif BACKBONE == 'squeezenet1_0':
    net = squeezenet1_0(num_classes=2).to(device)
elif BACKBONE == 'squeezenet1_1':
    net = squeezenet1_1(num_classes=2).to(device)
elif BACKBONE == 'mnasnet0_5':
    net = mnasnet0_5(num_classes=2).to(device)
elif BACKBONE == 'mnasnet1_0':
    net = mnasnet1_0(num_classes=2).to(device)
elif BACKBONE == 'mobilenet_v1':
    net = MobileNetV1(num_classes=2).to(device)
else:
    raise Exception('unknow backbone: {}'.format(BACKBONE))

for m in net.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0, 0.1)
        m.bias.data.zero_()


# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss().to(device)

# ============================ step 4/5 优化器 ============================
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)  # , momentum=0.9
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# ============================ step 5/5 训练 ============================
train_curve = list()
valid_curve = list()
test_curve = list()

iter_count = 0

# 构建 SummaryWriter
SAVE_FIELD = '{}-{}'.format(SAVE_FIELD, BACKBONE, )
writer = SummaryWriter(log_dir=os.path.join('logs', SAVE_FIELD))

for epoch in range(MAX_EPOCH):

    total_loss = 0.
    correct = 0.
    total = 0.

    net.train()
    lr = scheduler.get_lr()[-1]
    for i, data in enumerate(train_loader):

        iter_count += 1

        # forward
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 返回单个图片的可视化方法
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # img_tensor = inputs[0, ...]  # C H W
        # img = transform_invert(img_tensor, train_transform)
        # img = np.array(img)
        # cv2.imshow('show', img)
        # cv2.waitKey()
        # continue
        # plt.imshow(img)
        # plt.show()
        # plt.pause(0.5)
        # plt.close()
        # continue
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        outputs = net(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # update weights
        optimizer.step()

        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)
        nsample_step = labels.size(0)  # 本次step有多少个样本
        total += nsample_step  # 本次epoch样本累计
        correct_step = (predicted == labels).squeeze().sum().cpu().numpy()  # 本次step中的正确计数
        correct += correct_step  # 本次epoch正确计数

        # 打印训练信息
        loss_step = loss.item()
        total_loss += loss_step
        train_curve.append(loss_step)
        if (i+1) % log_interval == 0:
            text = "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, i+1, len(train_loader), loss_step, correct_step / nsample_step)
            text += " Lr:{:.4f}".format(lr)
            print(text)

        writer.add_scalar("StepLoss", loss_step, iter_count)
        writer.add_scalar("StepAccuracy", correct_step / nsample_step, iter_count)

    writer.add_scalars("Loss", {"Train": total_loss / total}, epoch)
    writer.add_scalars("Accuracy", {"Train": correct / total}, epoch)

    for name, param in net.named_parameters():
        writer.add_histogram(name + '_grad', param.grad, epoch)
        writer.add_histogram(name + '_data', param, epoch)

    writer.add_scalar("lr", lr, epoch)

    scheduler.step()  # 更新学习率

    # validate the model
    if (epoch+1) % val_interval == 0:
        correct_val = 0.
        loss_val = 0.
        total_val = 0
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                nsample_step = labels.size(0)
                total_val += nsample_step
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
                loss_val += loss.item()

            loss_val_mean = loss_val / total_val
            acc_val = correct_val / total_val
            valid_curve.append(loss_val_mean)
            text = "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, acc_val)
            print(text)

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Valid": loss_val_mean}, epoch)
            writer.add_scalars("Accuracy", {"Valid": acc_val}, epoch)

    # test the model
    if (epoch+1) % test_interval == 0:
        correct_val = 0.
        loss_val = 0.
        total_val = 0
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                nsample_step = labels.size(0)
                total_val += nsample_step
                correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
                loss_val += loss.item()

            loss_val_mean = loss_val / total_val
            acc_val = correct_val / total_val
            test_curve.append(loss_val_mean)
            text = "Test:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch, MAX_EPOCH, j+1, len(test_loader), loss_val_mean, acc_val)
            print(text)

            # 记录数据，保存于event file
            writer.add_scalars("Loss", {"Test": loss_val_mean}, epoch)
            writer.add_scalars("Accuracy", {"Test": acc_val}, epoch)

        if acc_val > 0.95:
            save_folder = os.path.join(MODEL_SAVE_PATH, SAVE_FIELD)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_folder = os.path.join(MODEL_SAVE_PATH, SAVE_FIELD, 'acc{:.4f}-loss{:.4f}-epoch{}.pth'.format(acc_val, loss_val_mean, epoch + 1))
            torch.save(net.state_dict(), save_folder)

save_folder = os.path.join(MODEL_SAVE_PATH, SAVE_FIELD, 'final.pth')
torch.save(net.state_dict(), save_folder)


train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()

