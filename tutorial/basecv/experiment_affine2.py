from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


img_path = "/home/tangni/res/himo1_128.jpg"
img = Image.open(img_path)
imgw, imgh = img.size
img_torch = transforms.ToTensor()(img)

plt.imshow(img_torch.numpy().transpose(1,2,0))
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import torch

offset_w = 5
offset_h = 10

theta = np.array([
    [1,0,offset_w],
    [0,1,offset_h]
])
# 变换1：可以实现缩放/旋转，这里为 [[1,0],[0,1]] 保存图片不变
t1 = theta[:,[0,1]]
# 变换2：可以实现平移
t2 = theta[:,[2]]

_, h, w = img_torch.size()
new_img_torch = torch.zeros_like(img_torch, dtype=torch.float)
for x in range(w):
    for y in range(h):
        pos = np.array([[x], [y]])
        npos = t1@pos+t2
        nx, ny = npos[0][0], npos[1][0]
        if 0<=nx<w and 0<=ny<h:
            new_img_torch[:,ny,nx] = img_torch[:,y,x]
plt.imshow(new_img_torch.numpy().transpose(1,2,0))
plt.show()



from torch.nn import functional as F

theta = torch.tensor([
    [1,0,-1],
    [0,1,-1]
], dtype=torch.float)
grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
output = F.grid_sample(img_torch.unsqueeze(0), grid)
new_img_torch = output[0]
plt.imshow(new_img_torch.numpy().transpose(1,2,0))
plt.show()