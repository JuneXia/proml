import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        self.model_init_conv_block = nn.Sequential(*model)

        in_features = out_features
        model = []
        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        self.model_down_sampling = nn.Sequential(*model)

        model = []
        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        self.model_residual_block = nn.Sequential(*model)

        model = []
        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        self.model_up_sampling = nn.Sequential(*model)

        model = []
        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]
        self.model_output_layer = nn.Sequential(*model)

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        # return self.model(x)
        x = self.model_init_conv_block(x)
        x = self.model_down_sampling(x)
        x = self.model_residual_block(x)
        x = self.model_up_sampling(x)
        x = self.model_output_layer(x)

        return x



##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_block1 = nn.Sequential(*discriminator_block(channels, 64, normalize=False))
        self.model_block2 = nn.Sequential(*discriminator_block(64, 128, normalize=False))
        self.model_block3 = nn.Sequential(*discriminator_block(128, 256, normalize=False))
        self.model_block4 = nn.Sequential(*discriminator_block(256, 512, normalize=False))

        model = [nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(512, 1, 4, padding=1)]
        self.model_output = nn.Sequential(*model)

        # self.model = nn.Sequential(
        #     *discriminator_block(channels, 64, normalize=False),
        #     *discriminator_block(64, 128),
        #     *discriminator_block(128, 256),
        #     *discriminator_block(256, 512),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(512, 1, 4, padding=1)
        # )

    def forward(self, img):
        # return self.model(img)
        img = self.model_block1(img)
        img = self.model_block2(img)
        img = self.model_block3(img)
        img = self.model_block4(img)
        img = self.model_output(img)
        return img

