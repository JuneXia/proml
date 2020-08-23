import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn.functional as F
import cv2


def gen_grid_range_fu1_1(o_dims, batch_size=1):
    height, width = o_dims

    x = np.linspace(-1.0, 1.0, width, endpoint=False, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, endpoint=False, dtype=np.float32)
    # x = np.linspace(0, width, width, endpoint=False)
    # y = np.linspace(0, height, height, endpoint=False)

    x_table, y_table = np.meshgrid(x, y)
    grid_id = np.array((x_table, y_table))
    grid_id = torch.tensor(grid_id)
    # grid_id = grid_id.permute((1, 2, 0))
    grid_id = grid_id.repeat((batch_size, 1, 1, 1))
    return grid_id


def gen_grid(o_dims, batch_size=1):
    height, width = o_dims

    x = np.arange(0, width, dtype=np.float32)
    y = np.arange(0, height, dtype=np.float32)

    x_table, y_table = np.meshgrid(x, y)
    grid_id = np.array((x_table, y_table))
    grid_id = torch.tensor(grid_id)
    # grid_id = grid_id.permute((1, 2, 0))
    grid_id = grid_id.repeat((batch_size, 1, 1, 1))

    grid_id = grid_id.requires_grad_(False)
    return grid_id


def init_grid_id(input):
    bs, c, h, w = input.shape
    xx = torch.arange(0, w).view(1, -1).repeat(h, 1)
    yy = torch.arange(0, h).view(-1, 1).repeat(1, w)
    xx = xx.view(1, 1, h, w).repeat(bs, 1, 1, 1)
    yy = yy.view(1, 1, h, w).repeat(bs, 1, 1, 1)
    grid_id = torch.cat((xx, yy), 1).float().requires_grad_(False)

    return grid_id


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, grid_size=(3, 3), cfg=None):
        super(GeneratorUNet, self).__init__()

        self.grid_size = grid_size

        # TODO：grid_template 放在这里创建似乎并不太合适。

        grid_template = cv2.imread(cfg['grid_template'])  # TODO：因为这里处理的mnist读取灰度图
        # grid_template = grid_template - 70
        if len(grid_template.shape) >= 3 and in_channels == 1:
            grid_template = cv2.cvtColor(grid_template, cv2.COLOR_BGR2GRAY)
        grid_template = cv2.resize(grid_template, (28, 28), interpolation=cv2.INTER_LINEAR)
        grid_template = grid_template.reshape((28, 28, 1))
        grid_template = grid_template/255.0
        grid_template = (grid_template - 0.5) / 0.5
        grid_template = torch.FloatTensor(grid_template).unsqueeze(0)
        grid_template = grid_template.permute((0, 3, 1, 2))
        self.grid_template = grid_template.to(cfg['device'])

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        # self.down4 = UNetDown(256, 512, dropout=0.5)
        # self.down5 = UNetDown(512, 512, dropout=0.5)
        # self.down6 = UNetDown(512, 512, dropout=0.5)
        # self.down7 = UNetDown(512, 512, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # self.up1 = UNetUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        # self.up4 = UNetUp(1024, 512, dropout=0.5)
        # self.up5 = UNetUp(1024, 256)
        # self.up6 = UNetUp(512, 128)
        # self.up7 = UNetUp(256, 64)

        self.final = nn.Conv2d(128, 2, 1)

        # self.final = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, out_channels, 4, padding=1),
        #     nn.Tanh(),
        # )

    def forward_ok(self, inputa, grid_base):
        B, C, H, W = inputa.shape

        # U-Net generator with skip connections from encoder to decoder
        x = self.down1(inputa)
        x = self.down2(x)
        # x = self.down3(x)
        sparse_grid_offset = self.final(x)

        # sparse_grid = grid_base + sparse_grid_offset

        dense_grid_offset = F.interpolate(sparse_grid_offset, size=(H, W), scale_factor=None, mode='bilinear', align_corners=True)

        dense_grid = grid_base + dense_grid_offset

        vgrid = torch.clamp(dense_grid, 0, max(W - 1, 1))  # constraint of predict flow

        scale = 1
        vgrid[:, 0, :, :] = 2 * scale * vgrid[:, 0, :, :] / max(W - 1, 1) - scale
        vgrid[:, 1, :, :] = 2 * scale * vgrid[:, 1, :, :] / max(H - 1, 1) - scale

        vgrid = vgrid.permute((0, 2, 3, 1))
        fakeb = F.grid_sample(inputa, vgrid, mode='bilinear', padding_mode='zeros')

        return fakeb, dense_grid_offset

    def forward(self, inputa, sparse_grid_base, dense_grid_base):
        B, C, H, W = inputa.shape

        # U-Net generator with skip connections from encoder to decoder
        x = self.down1(inputa)
        x = self.down2(x)
        # x = self.down3(x)
        x = self.final(x)

        sparse_grid_offset = F.adaptive_avg_pool2d(x, output_size=self.grid_size)

        sparse_grid_offset = sparse_grid_base + sparse_grid_offset

        dense_grid_offset = F.interpolate(sparse_grid_offset, size=(H, W), scale_factor=None, mode='bilinear', align_corners=True)

        dense_grid = dense_grid_base + dense_grid_offset

        vgrid = torch.clamp(dense_grid, 0, max(W - 1, 1))  # constraint of predict flow

        scale = 1
        vgrid[:, 0, :, :] = 2 * scale * vgrid[:, 0, :, :] / max(W - 1, 1) - scale
        vgrid[:, 1, :, :] = 2 * scale * vgrid[:, 1, :, :] / max(H - 1, 1) - scale

        vgrid = vgrid.permute((0, 2, 3, 1))
        fakeb = F.grid_sample(inputa, vgrid, mode='bilinear', padding_mode='zeros')

        # ********************************************************
        grid_template = self.grid_template.repeat((B, 1, 1, 1))
        grid_template_warp = F.grid_sample(grid_template, vgrid, mode='bilinear', padding_mode='zeros')
        # ********************************************************

        return fakeb, sparse_grid_offset, dense_grid_offset, grid_template_warp

    def forward_不行(self, inputa, grid_base):
        B, C, H, W = inputa.shape

        # U-Net generator with skip connections from encoder to decoder
        x = self.down1(inputa)
        x = self.down2(x)
        x = self.down3(x)
        sparse_grid_offset = self.final(x)

        sparse_grid = grid_base + sparse_grid_offset

        dense_grid = F.interpolate(sparse_grid, size=(H, W), scale_factor=None, mode='bilinear', align_corners=True)

        # dense_grid = grid_base + dense_grid_offset

        vgrid = torch.clamp(dense_grid, 0, max(W - 1, 1))  # constraint of predict flow

        scale = 1
        vgrid[:, 0, :, :] = 2 * scale * vgrid[:, 0, :, :] / max(W - 1, 1) - scale
        vgrid[:, 1, :, :] = 2 * scale * vgrid[:, 1, :, :] / max(H - 1, 1) - scale

        vgrid = vgrid.permute((0, 2, 3, 1))
        fakeb = F.grid_sample(inputa, vgrid, mode='bilinear', padding_mode='zeros')

        return fakeb, sparse_grid_offset


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # self.model = nn.Sequential(
        #     *discriminator_block(in_channels * 2, 64, normalization=False),
        #     *discriminator_block(64, 128),
        #     *discriminator_block(128, 256),
        #     *discriminator_block(256, 512),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(512, 1, 4, padding=1, bias=False)
        # )

        self.layer1 = nn.Sequential(*discriminator_block(in_channels * 2, 64, normalization=False))
        self.layer2 = nn.Sequential(*discriminator_block(64, 128))
        self.layer3 = nn.Sequential(*discriminator_block(128, 256))
        # self.layer4 = nn.Sequential(*discriminator_block(256, 512))
        self.zero_pad = nn.ZeroPad2d((1, 0, 1, 0))
        # self.final = nn.Conv2d(512, 1, 4, padding=1, bias=False)  # 原版
        self.final = nn.Conv2d(256, 1, 3, padding=1, bias=False)  # TODO：modify

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)

        # output = self.model(img_input)

        output = self.layer1(img_input)
        output = self.layer2(output)
        output = self.layer3(output)
        # output = self.layer4(output)
        output = self.zero_pad(output)
        output = self.final(output)

        return output


# from torchsummary import summary
if __name__ == '__main__':
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        torch.cuda.set_device(2)

    generator = GeneratorUNet()
    discriminator = Discriminator()
    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    summary(generator, [3, 256, 256])
    summary(discriminator, [[3, 256, 256], [3, 256, 256]])