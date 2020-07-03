import glob
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import cv2


class CMPFacade(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform

        self.images = sorted(glob.glob(os.path.join(root, "*.jpg")))  # real-image
        self.pixwises = sorted(glob.glob(os.path.join(root, "*.png")))  # mask-label, condition
        assert len(self.images) == len(self.pixwises)

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        pixw = Image.open(self.pixwises[index])
        # print(self.images[index], self.pixwises[index])

        img = np.array(img)
        pixw = np.array(pixw)
        if len(pixw.shape) == 2:
            pixw = cv2.cvtColor(pixw, cv2.COLOR_GRAY2RGB)

        if np.random.random() < 0.5:
            img = img[:, ::-1, :]
            pixw = pixw[:, ::-1, :]

        img = Image.fromarray(img, "RGB")
        pixw = Image.fromarray(pixw, "RGB")

        img = self.transform(img)
        pixw = self.transform(pixw)

        return {"A": pixw, "B": img}

    def __len__(self):
        return len(self.images)


from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tutorial.tools import common_tools
if __name__ == '__main__':
    data_path = '/home/tangni/res/CMP_facade/CMP_facade_DB_base/base'
    img_height, img_width = 256, 256
    batch_size = 1
    n_cpu = 1

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataloader = DataLoader(
        CMPFacade(data_path, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
    )

    for i, batch in enumerate(dataloader):
        cond_A = batch["A"]
        real_B = batch["B"]

        cond_A = cond_A[0, ...]
        real_B = real_B[0, ...]
        tmpimg = common_tools.normalize_invert(batch["A"], transform)
        img_A = common_tools.transform_invert(cond_A, transform)
        img_B = common_tools.transform_invert(real_B, transform)
        img_A = np.array(img_A)*20
        img_B = np.array(img_B)
        cv2.imshow('img_A', img_A)
        cv2.imshow('img_B', img_B)
        cv2.waitKey(500)

