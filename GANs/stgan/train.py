import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
# from GANs.gan.model import Generator
# from GANs.gan.model import Discriminator

from config import cfg_net
from datasets import MnistPairDataset
from model import GeneratorUNet as Generator
from model import Discriminator

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(in_channels=img_shape[0], out_channels=opt.latent_dim)
discriminator = Discriminator(in_channels=img_shape[0])

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

transformer = transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

# Configure data loader

dataset = MnistPairDataset(cfg_net, transforms=transformer)
# dataset = datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transformer,
#     )
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):
        imagea = data['imagea']
        imageb = data['imageb']

        # Adversarial ground truths
        valid = Variable(Tensor(imagea.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imagea.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imagea.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imagea.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fakeb = generator(imagea)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(imagea, fakeb), valid)  # 我们希望Generator生成的图片更加逼近真实样本，所以训练Generator时的ground-truth label为1

        g_loss.backward()  # 求导
        optimizer_G.step()  # 更新参数

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()  # 梯度清零

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(imagea, imageb), valid)  # 真实数据的ground-truth label是1
        fake_loss = adversarial_loss(discriminator(fakeb.detach(), imageb), fake)  # 假数据的ground-truth label是0
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()  # 求导
        optimizer_D.step()  # 参数更新

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(fakeb.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)



