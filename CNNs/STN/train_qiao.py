# -*- coding: utf-8 -*-
# @Time    : 2020/4/13
# @Author  : Lafe
# @Email   : wangdh8088@163.com
# @File    : generate_f32.py

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.utils.data as data
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from easydict import EasyDict as edict
import os
import sys
import os.path as osp
import  time
import pdb
sys.path.insert(0, '.')
try:
    from dlib_facekeys.landmark_loss_v2 import LandMarkLoss
except Exception as e:
    print(" * error ", e)

def nothing(img):
    return img

def showrgb(img):
    if isinstance(img, list):
        img = np.concatenate(img, axis=1)

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imgrgb)
    plt.show()


class Dataset():
    def __init__(self, root, apath, bpath, trans=None, opt=None):
        abs_apath = osp.join(root, apath)
        abs_bpath = osp.join(root, bpath)
        self.a_ps = [osp.join(abs_apath, i) for i in os.listdir(abs_apath)]
        self.b_ps = [osp.join(abs_bpath, i) for i in os.listdir(abs_bpath)]
        self.trans = trans if (trans is not None) else nothing
        self.opt = opt
        self.flag_saveinmem = opt.flag_saveinmem
        if self.flag_saveinmem == True:
            self.a_imgs = []
            self.b_imgs = []
            for aname, bname in zip(self.a_ps, self.b_ps):
                self.a_imgs.append(cv2.imread(aname))
                self.b_imgs.append(cv2.imread(bname))

        if opt.get('use_dlibmap') is not None and opt.use_dlibmap == True:
            self.landm = LandMarkLoss(opt.dlib_path)

    def _dilbmap(self, a, b):

        try:
            landa = self.landm.infer_landmark(a)  # (1, 68)
            landb = self.landm.infer_landmark(b)  # (1, 68)

            landa = landa[0].tolist()
            landb = landb[0].tolist()

            dlib_a = self.landm.create_dlibmap(landa, a)
            dlib_b = self.landm.create_dlibmap(landb, b)

            dlib_a = torch.from_numpy(dlib_a.transpose((2, 0, 1))).float()
            dlib_b = torch.from_numpy(dlib_b.transpose((2, 0, 1))).float()
            return dict(dlib_a=dlib_a, dlib_b=dlib_b)

        except:
            # detect no face
            dlib_a = np.zeros_like(a, dtype=np.uint8)
            dlib_b = np.zeros_like(b, dtype=np.uint8)
            dlib_a = TF.to_tensor(dlib_a)
            dlib_b = TF.to_tensor(dlib_b)
            return dict(dlib_a=dlib_a, dlib_b=dlib_b)

    def __getitem__(self, index):
        ### input A (label maps)
        aname = self.a_ps[index]
        bname = self.b_ps[index]

        if self.flag_saveinmem == True:
            raw_a = self.a_imgs[index]
            raw_b = self.b_imgs[index]
        else:
            raw_a = cv2.imread(aname)
            raw_b = cv2.imread(bname)

        a = self.trans(raw_a, self.opt)
        b = self.trans(raw_b, self.opt)
        A_tensor = TF.to_tensor(a)
        B_tensor = TF.to_tensor(b)

        input_dict = dict(
            a=A_tensor, b=B_tensor, aname=self.a_ps[index], raw_a=raw_a, raw_b=raw_b
        )

        if self.opt.get('use_dlibmap') is not None and self.opt.use_dlibmap == True:
            dlib_dict = self._dilbmap(a, b)
            input_dict.update(dlib_dict)
        return input_dict

    def __len__(self):
        return len(self.a_ps)

    def name(self):
        return 'datasets'


class Stn(nn.Module):
    def __init__(self, in_channel=3, inshape=26):
        super(Stn, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channel, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # N = (W - F + 2p) / S  + 1
        loc_shape = int(((inshape - 7 + 1)/2 - 5 + 1)/2)
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * loc_shape * loc_shape, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        b, c, h, w = xs.shape
        # xs = xs.view(-1, 10 * 3 * 3)
        xs = xs.view(-1, c * h * w)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        self.theta_tmp = theta.data.cpu().numpy()
        # print(theta)
        grid = F.affine_grid(theta, x.size()) # affine with theta
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        return x

class StnLandmark(Stn):
    def forward(self, x):
        pass

def adjust_learning_rate(optimizer, step, decay_rate=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def dlibloss(sample, landm, net, device):

    a = sample['dlib_a'].to(device)
    b = sample['dlib_b'].to(device)
    with torch.no_grad():
        theta = torch.from_numpy(net.theta_tmp).to(device)
        grid = F.affine_grid(theta, a.size()) # affine with theta
        warp_a = F.grid_sample(a, grid)
    warp_a = torch.ceil(warp_a)
    print('***' * 100)
    print(np.unique(a.cpu().numpy()))
    print('***' * 100)

    loss = landm(warp_a, b).to(device)

    return loss

def train(opt):
    print(' ****** start train ******')
    for k, v in opt.items():
        print(" * {} : {}".format(k, v))
    inputshape = opt.trans_param.wh[0]
    channel = opt.channel
    bs = opt.bs
    base_lr = opt.base_lr

    total_epoch = opt.total_epoch
    device = opt.device
    #  datasets
    datas = Dataset(opt.root, apath=opt.ap, bpath=opt.bp, trans=opt.trans, opt=opt.trans_param)
    datas_init = datas[0]
    dataloader = torch.utils.data.DataLoader(datas, bs, shuffle=True, num_workers=opt.works, drop_last=True)
    #  models
    net = Stn(in_channel=channel, inshape=inputshape).to(device)
    net.train()
    #  optmizer
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    #  loss init
    # landmloss = LandMarkLoss(opt.trans_param.dlib_path)
    print(' ****** start train .....')
    for epoch in range(total_epoch):
        for step, sample in enumerate(dataloader):
            s = time.time()

            optimizer.zero_grad()
            a = sample['a'].to(device)
            b = sample['b'].to(device)
            out = net(a)
            loss = F.l1_loss(b, out)
            # loss = dlibloss(sample, landmloss, net, device)
            # loss = loss_func(out, b)
            loss.backward()
            optimizer.step()
            print('Train step: {}/{} {}/{} Loss: {:.6f}  lr: {}  time: {:.3f}'.format(
                step + epoch*bs, total_epoch*bs, epoch, total_epoch, loss.item(), optimizer.param_groups[0]['lr'],
                time.time() - s
            ))
        if epoch % opt.adj_lr == 0 and epoch != 0:
            adjust_learning_rate(optimizer, epoch)

    torch.save(net.state_dict(), 'stn.pth')
    print(' * train end ')

def test(opt):
    print(' ****** start test ******')
    for k, v in opt.items():
        print(" * {} : {}".format(k, v))

    inputshape = opt.trans_param.wh[0]
    channel = opt.channel
    bs = opt.bs
    base_lr = opt.base_lr

    total_epoch = opt.total_epoch
    device = opt.device
    datas = Dataset(opt.root, apath=opt.ap, bpath=opt.bp, trans=opt.trans, opt=opt.trans_param)
    datas_init = datas[0]
    dataloader = torch.utils.data.DataLoader(datas, 1, shuffle=False, num_workers=opt.works, drop_last=False)

    pretrain = torch.load('stn.pth')
    net = Stn(in_channel=channel, inshape=inputshape).to(device)
    net.load_state_dict(pretrain)
    net.eval()
    net = net.to(device)



    os.makedirs(opt.savep, exist_ok=True)
    with torch.no_grad():
        for step, sample in enumerate(tqdm(dataloader)):
            a = sample['a'].to(device)
            b = sample['b'].to(device)
            aname = sample['aname']
            pred = net(a)

            pred = pred.cpu()
            img = torchvision.utils.make_grid(pred)
            img = img.numpy().transpose((1, 2, 0)) * 255

            a = a.cpu()
            a = torchvision.utils.make_grid(a)
            a = a.numpy().transpose((1, 2, 0)) * 255
            b = b.cpu()
            b = torchvision.utils.make_grid(b)
            b = b.numpy().transpose((1, 2, 0)) * 255

            save_name = osp.basename(aname[0])
            cv2.imwrite(osp.join(opt.savep, save_name.replace('.', '_1_pred.')), img)
            cv2.imwrite(osp.join(opt.savep, save_name.replace('.', '_3_a.')), a)
            cv2.imwrite(osp.join(opt.savep, save_name.replace('.', '_2_b.')), b)

def trans(img, opt):
    w, h = opt.wh
    img = cv2.resize(img, (w, h))
    # img = cv2.blur(img, (2, 2))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # x = cv2.Sobel(img,cv2.CV_16S,1,0)
    # y = cv2.Sobel(img,cv2.CV_16S,0,1)


    # absX = cv2.convertScaleAbs(x)   # 转回uint8
    # absY = cv2.convertScaleAbs(y)
    #
    # img = cv2.addWeighted(absX,0.5,absY,0.5,0)
    # img = np.concatenate([img[..., None], img[...,None], img[..., None]], axis=2)

    return img

def dlibmap(pred, b):

    return torch.tensor(99, dtype=torch.float, requires_grad=True)
    pass

if __name__ == '__main__':
    opt = edict(
        device='cpu',
        root='/Users/wdh/mt/test_images/23/test_latest',
        ap='images',
        bp='labels',
        inputshape=512,
        channel = 3,
        bs = 40,
        base_lr = 0.05,
        total_epoch = 10,
        works=2
    )

    opt2 = edict(
        device='cuda:3',
        root='/home/lafe/datasets/cert-link-5000-crop_warp_filtrate',
        # ap='train_A',
        # bp='train_B',
        ap='test_A',
        bp='test_B',
        channel = 3,
        bs = 3, #600,
        base_lr = 0.05,
        total_epoch = 5,
        works=40,
        adj_lr=10,

        #loss
        # loss_func=nn.L1Loss(),
        loss_func=dlibmap,
        # testa = 'test_A',
        testa = 'train_A',
        testb = 'test_B',
        savep = 'warp',

        trans=trans,
        trans_param=edict(
            flag_saveinmem=False,
            wh=[512, 512],

            use_dlibmap=False,
            dlib_path='dlib_facekeys/shape_predictor_68_face_landmarks.dat',
        )
    )


    opt250 = edict(
        device='cuda:1',
        root="/home/lafe/mt_data/a2b/cert-link-5000-crop_warp",
        ap='test_A',
        bp='test_B',

        channel = 3,
        bs = 180,
        base_lr = 0.1,
        total_epoch = 100,
        works=1,
        adj_lr=20,

        # testa = 'test_A',
        testa = 'test_A',
        testb = 'test_A',
        savep = 'warp',
        trans=trans,
        trans_param=edict(
            flag_saveinmem=True,
            wh=[512, 512],

            use_dlibmap=False,
            dlib_path='dlib_facekeys/shape_predictor_68_face_landmarks.dat',
        )

    )
    if torch.cuda.device_count() != 0:
        default_opt = opt250
    else:
        default_opt = opt

    train(default_opt)
    # test(default_opt)

