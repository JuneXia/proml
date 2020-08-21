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
import model
from model import Discriminator
import losses

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

opt.batch_size = 128
grid_size = cfg_net['grid_size']

img_shape = (opt.channels, opt.img_size, opt.img_size)

# cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loss function
# adversarial_loss = torch.nn.BCELoss()
criterion_GAN_loss = torch.nn.MSELoss()
grid_regular_loss = losses.GridRegular(grad_lambda=2)
l1_loss = torch.nn.L1Loss()
l1_loss_lambda = 10

# Initialize generator and discriminator
generator = Generator(in_channels=img_shape[0], out_channels=opt.latent_dim, grid_size=grid_size)
generator2 = Generator(in_channels=img_shape[0], out_channels=opt.latent_dim, grid_size=grid_size)
discriminator = Discriminator(in_channels=img_shape[0])

generator.cuda(device)
generator2.cuda(device)
discriminator.cuda(device)
criterion_GAN_loss.cuda(device)

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
optimizer_G2 = torch.optim.Adam(generator2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.FloatTensor

# ----------
#  Training
# ----------
if __name__ == '__main__2':
    for epoch in range(opt.n_epochs):
        for i, data in enumerate(dataloader):
            imagea = data['imagea'].to(device)
            imageb = data['imageb'].to(device)

            dense_grid_size = (imagea.shape[2], imagea.shape[3])
            dense_grid_base = model.init_grid_id(imagea).to(device)

            sparse_grid_base = Variable(Tensor(imagea.size(0), 2, grid_size[0], grid_size[1]).fill_(0.0), requires_grad=False).to(device)
            # sparse_grid_base = model.init_grid_id(imagea).to(device)

            # grid_size = (3, 3)
            # grid_base = model.init_grid_id(Tensor(imagea.size(0), 2, 3, 3)).to(device)

            # grid_base = model.gen_grid(grid_size, batch_size=imagea.shape[0]).to(device)

            # TODO: grid_base 只要初始化时创建一次即可（但要注意最后一个batch的batch_size大小问题），没必要每次都创建

            # Adversarial ground truths
            # valid = Variable(Tensor(imagea.size(0), 1).fill_(1.0), requires_grad=False).to(device)
            # fake = Variable(Tensor(imagea.size(0), 1).fill_(0.0), requires_grad=False).to(device)

            # Configure input
            real_imgs = Variable(imagea.type(Tensor)).to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            # z = Variable(Tensor(np.random.normal(0, 1, (imagea.shape[0], opt.latent_dim)))).to(device)

            # Generate a batch of images
            fakeb, sparse_grid_offset = generator(imagea, sparse_grid_base, dense_grid_base)

            # Loss measures generator's ability to fool the discriminator
            pred_fake = discriminator(imagea, fakeb)
            valid = Variable(
                Tensor(imagea.size(0), pred_fake.shape[1], pred_fake.shape[2], pred_fake.shape[3]).fill_(1.0),
                requires_grad=False).to(device)
            fake = Variable(
                Tensor(imagea.size(0), pred_fake.shape[1], pred_fake.shape[2], pred_fake.shape[3]).fill_(0.0),
                requires_grad=False).to(device)
            gan_loss = criterion_GAN_loss(pred_fake,
                                          valid)  # 我们希望Generator生成的图片更加逼近真实样本，所以训练Generator时的ground-truth label为1

            grid_loss = grid_regular_loss(sparse_grid_offset)
            pixel_loss = l1_loss(fakeb, imageb) * l1_loss_lambda

            g_loss = gan_loss + grid_loss + pixel_loss

            g_loss.backward()  # 求导
            optimizer_G.step()  # 更新参数

            # -------------------------------------
            imagea = data['imagea'].to(device)
            imageb = data['imageb'].to(device)

            dense_grid_size = (imagea.shape[2], imagea.shape[3])
            dense_grid_base = model.init_grid_id(imagea).to(device)

            sparse_grid_base = Variable(Tensor(imagea.size(0), 2, grid_size[0], grid_size[1]).fill_(0.0), requires_grad=False).to(device)

            optimizer_G2.zero_grad()
            fakeb, sparse_grid_offset = generator2(imagea, sparse_grid_offset.detach(), dense_grid_base)
            pred_fake = discriminator(imagea, fakeb)
            valid = Variable(
                Tensor(imagea.size(0), pred_fake.shape[1], pred_fake.shape[2], pred_fake.shape[3]).fill_(1.0),
                requires_grad=False).to(device)
            fake = Variable(
                Tensor(imagea.size(0), pred_fake.shape[1], pred_fake.shape[2], pred_fake.shape[3]).fill_(0.0),
                requires_grad=False).to(device)
            gan_loss = criterion_GAN_loss(pred_fake,
                                          valid)  # 我们希望Generator生成的图片更加逼近真实样本，所以训练Generator时的ground-truth label为1

            grid_loss = grid_regular_loss(sparse_grid_offset)
            pixel_loss = l1_loss(fakeb, imageb) * l1_loss_lambda

            g_loss = gan_loss + grid_loss + pixel_loss

            g_loss.backward()  # 求导
            optimizer_G.step()  # 更新参数

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()  # 梯度清零

            # Measure discriminator's ability to classify real from generated samples
            real_loss = criterion_GAN_loss(discriminator(imagea, imageb), valid)  # 真实数据的ground-truth label是1
            # fake_loss = criterion_GAN_loss(discriminator(fakeb.detach(), imageb), fake)  # 假数据的ground-truth label是0
            fake_loss = criterion_GAN_loss(discriminator(imagea, fakeb.detach()), fake)  # 假数据的ground-truth label是0
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()  # 求导
            optimizer_D.step()  # 参数更新

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ganls: %f] [gridls: %f] [pixls: %f] [drels: %f] [dfakls: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), gan_loss.item(),
                   grid_loss.item(), pixel_loss.item(), real_loss.item(), fake_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                sample_img = torch.cat((imagea.data, imageb.data, fakeb.data), -2)
                save_image(sample_img.data[:25], "images/%d.png" % batches_done, nrow=10, normalize=True)


