import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


if __name__ == '__main__1':  # 关于 make_grid 的 tenosr 拼接维度以及nrow参数的实验
    lena = scipy.misc.face()
    img = transforms.ToTensor()(lena)
    print('\n原始的 img.shape: \n', img.shape)

    img = torch.unsqueeze(img, dim=0)
    print('\n补充batch维度后的 img.shape: \n', img.shape)

    img = torch.cat((img, img, img), 0)
    print('\n拼成batch_size为3后的 img.shape: \n', img.shape)

    # 多制造几组tensor, 每组tensor再调整下亮度，用于区分不同的组的数据
    imglist = [img * 0.2, img * 0.5, img * 0.8, img.clone().fill_(1) * 0.5, img * 1.2]

    # 将多组tensor在batch维度上去拼接：
    tensor1 = torch.cat(imglist, 0)
    print('\n将各tensor在batch维度拼接后的 tensor.shape: \n', tensor1.shape)
    tensor1 = make_grid(tensor1, nrow=8, padding=100)
    print('\nmake_grid 后的 tensor.shape: \n', tensor1.shape)

    show(tensor1)


    # 将多组tensor在height维度上去拼接:
    tensor2 = torch.cat(imglist, -2)
    print('\n将各tensor在height维度拼接后的 tensor.shape: \n', tensor2.shape)
    tensor2 = make_grid(tensor2, nrow=8, padding=100)
    print('\nmake_grid 后的 tensor.shape: \n', tensor2.shape)

    show(tensor2)


if __name__ == '__main__2':  # 关于 make_grid 的 normalize、range、scale_each 等参数的实验
    lena = scipy.misc.face()
    img = transforms.ToTensor()(lena)

    # 多制造几组tensor, 每组tensor再调整下亮度，用于区分不同的组的数据
    imglist = [img * 0.2, img * 0.5, img * 0.8, img.clone().fill_(1) * 0.5, img * 1.2]

    tensor1 = make_grid(imglist, padding=100, nrow=3)
    show(tensor1)

    tensor2 = make_grid(imglist, padding=100, nrow=3, normalize=True)
    show(tensor2)

    tensor3 = make_grid(imglist, padding=100, nrow=3, normalize=True, range=(0, 1))
    show(tensor3)

    tensor4 = make_grid(imglist, padding=100, nrow=3, normalize=True, range=(0, 0.5))
    show(tensor4)

    tensor5 = make_grid(imglist, padding=100, nrow=3, normalize=True, scale_each=True)
    show(tensor5)

    tensor6 = make_grid(imglist, padding=100, nrow=3, normalize=True, range=(0, 0.5), scale_each=True)
    show(tensor6)


from torchvision.utils import save_image
import cv2
if __name__ == '__main__':  # 关于 torchvision.utils.save_image 的实验
    lena = scipy.misc.face()
    img = transforms.ToTensor()(lena)
    img = torch.unsqueeze(img, dim=0)
    img = torch.cat((img, img, img), 0)

    # 多制造几组tensor, 每组tensor再调整下亮度，用于区分不同的组的数据
    imglist = [img * 0.2, img * 0.5, img * 0.8, img.clone().fill_(1) * 0.5, img * 1.2]

    img_sample = torch.cat(imglist, -2)
    tensor1 = make_grid(img_sample, nrow=5, normalize=True, padding=100, pad_value=255)
    show(tensor1)

    save_image(img_sample, "images.png", nrow=5, normalize=True, padding=100, pad_value=255)

    img = cv2.imread("images.png")
    # img = cv2.resize(img, None, fx=0.2, fy=0.2)
    plt.imshow(img, interpolation='nearest')
    plt.show()

