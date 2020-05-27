import torch
import torch.nn as nn
from torch.functional import F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.mobilenet import mobilenet_v2
from libml.models.mobilenet import MobileNetV1
import dataset
from config import cfg_net
from libml.utils import tools
from PIL import Image
import numpy as np
import cv2



BATCH_SIZE = cfg_net['batch_size']
DEVICE = 'cpu'  # cfg_net['device']

test_dir = '/disk1/home/xiaj/res/face/maskface/Experiment/MAFA-test-images-mtcnn_align182x182_margin44/test-images-detected_face-classified'

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
input_size = 96
margin_scale = 182/160

valid_transform = transforms.Compose([
    transforms.Resize((int(input_size*margin_scale), int(input_size*margin_scale))),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# test_data = dataset.MaskFaceDataset(test_dir, transform=valid_transform)
#
# test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)

device = torch.device(DEVICE)  # cpu, cuda:0

# net = MobileNetV1(num_classes=2)  # .to(device)
net = mobilenet_v2(num_classes=2, width_mult=0.35, inverted_residual_setting=None, round_nearest=8)  # .to(device)

# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()  # .to(device)

# model_path = '/disk1/home/xiaj/dev/proml/maskface/save_model/20200313-041207-mobilenet_v2/acc0.9710-loss0.0034-epoch113.pth'
# model_path = '/disk1/home/xiaj/dev/proml/maskface/save_model/20200325-013239-mobilenet_v1/acc0.9551-loss0.0042-epoch29.pth'
# model_path = '/home/xiajun/dev/proml/maskface/save_model/20200325-021920-mobilenet_v1/acc0.9676-loss0.0037-epoch122.pth'
# model_path = '/disk1/home/xiaj/dev/proml/maskface/save_model/20200327-030712-mobilenet_v1/acc0.9602-loss0.0043-epoch128.pth'
# model_path = '/disk1/home/xiaj/dev/proml/maskface/save_model/20200327-052039-mobilenet_v1/acc0.9565-loss0.0043-epoch112.pth'
model_path = '/disk1/home/xiaj/dev/proml/maskface/save_model/20200402-235510-mobilenet_v2/acc0.9540-loss0.0043-epoch102.pth'

# model_path = '/home/xiajun/dev/proml/maskface/save_model/20200327-052039-mobilenet_v1/acc0.9565-loss0.0043-epoch112.pth'
net.load_state_dict(torch.load(model_path, map_location=device))

if False:
    net.eval()
    with torch.no_grad():
        correct_val = 0.
        loss_val = 0.
        total_val = 0
        for j, data in enumerate(test_loader):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            nsample_step = labels.size(0)
            total_val += nsample_step
            correct_val += (predicted == labels).squeeze().sum().cpu().numpy()
            loss_val += loss.item()

        loss_val_mean = loss_val / total_val
        acc_val = correct_val / total_val
        text = "Test:\t Loss: {:.4f} Acc:{:.2%}".format(loss_val_mean, acc_val)
        print(text)
elif True:
    net.eval()
    with torch.no_grad():
        impath = '/disk1/home/xiaj/res/face/maskface/Experiment/MAFA-test-images-mtcnn_align182x182_margin44/test-images-detected_face-classified/no/test_00000008.png'
        # impath = '/home/xiajun/res/face/maskface/Experiment/test-images/test_00000002.png'
        # image = Image.open(impath).convert('RGB')
        image = cv2.imread(impath)
        image = cv2.resize(image, (input_size, input_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = image / 255.0
        image[..., 0] = (image[..., 0] - image[..., 0].mean()) / image[..., 0].std()
        image[..., 1] = (image[..., 1] - image[..., 1].mean()) / image[..., 1].std()
        image[..., 2] = (image[..., 2] - image[..., 2].mean()) / image[..., 2].std()

        image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image)
        image = image.permute(0, 3, 1, 2)

        count = 0
        total_time = 0.0
        while True:
            count += 1
            t = float(cv2.getTickCount())
            outputs = net(image)
            outputs = F.softmax(outputs)
            total_time += (float(cv2.getTickCount()) - t) / cv2.getTickFrequency()
            print(outputs, '\t execute time: ', total_time/count)

    # 生成一个样本供网络前向传播 forward()
    example = torch.rand(1, 3, input_size, input_size)

    # 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
    traced_script_module = torch.jit.trace(net, example)

    count = 0
    total_time = 0.0
    while True:
        count += 1
        t = float(cv2.getTickCount())
        outputs = traced_script_module(torch.ones(1, 3, input_size, input_size))
        outputs = F.softmax(outputs)
        total_time += (float(cv2.getTickCount()) - t) / cv2.getTickFrequency()
        print(outputs, '\t execute time: ', total_time / count)

    traced_script_module.save("mobilenetv1_conv1x1_in96_netclip_eph112.pt")
