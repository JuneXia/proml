import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import cv2

save_dir = 'C:\\Users\\Administrator\\res\\mnist\\MNIST_IMG'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

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

batch_size = 1

transformer = transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                # transforms.Normalize([0.5], [0.5])
            ]
        )

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        # train=True,
        # download=True,
        transform=transformer,
    ),
    batch_size=batch_size,
    shuffle=True,
)

for i, (imgs, labs) in enumerate(dataloader):
    img = imgs[0].permute(1, 2, 0) * 255
    img = img.cpu().numpy().astype(np.uint8)
    lab = labs.cpu().numpy()[0]
    impath = os.path.join(save_dir, str(lab))
    if not os.path.exists(impath):
        os.makedirs(impath)
    cv2.imwrite("{}/{}.jpg".format(impath, i), img)
    if i % 200 == 0:
        print('{}/{}'.format(i, len(dataloader)))



