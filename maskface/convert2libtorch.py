import torch
from torch.functional import F
from torchvision.models.mobilenet import mobilenet_v2
# from libml.models.mobilenet import MobileNetV1
from config import cfg_net


# ====================== STEP 1: 加载训练好的模型==========================
BATCH_SIZE = cfg_net['batch_size']
DEVICE = 'cpu'  # cfg_net['device']

# 输出图片尺寸
input_size = 96

# 定义模型运行设备
device = torch.device(DEVICE)  # cpu, cuda:0

# 定义模型
# net = MobileNetV1(num_classes=2)  # .to(device)
net = mobilenet_v2(num_classes=2, width_mult=0.35, inverted_residual_setting=None, round_nearest=8)  # .to(device)

# 加载已经训练好的模型
model_path = '/disk1/home/xiaj/dev/proml/maskface/save_model/20200402-235510-mobilenet_v2/acc0.9540-loss0.0043-epoch102.pth'
# model_path = '/home/xiajun/dev/proml/maskface/save_model/20200327-052039-mobilenet_v1/acc0.9565-loss0.0043-epoch112.pth'
net.load_state_dict(torch.load(model_path, map_location=device))

# 设置为验证模型
net.eval()

# ====================== STEP 2: 使用torch.jit.trace做一次前向推理 ==========================
# 生成一个样本供网络前向传播 forward()
example = torch.rand(1, 3, input_size, input_size)

# 使用 torch.jit.trace 生成 torch.jit.ScriptModule 来跟踪
traced_script_module = torch.jit.trace(net, example)
outputs = traced_script_module(torch.ones(1, 3, input_size, input_size))
outputs = F.softmax(outputs)
print(outputs)

# ====================== STEP 2: 保存模型 ==========================
traced_script_module.save("model_name.pt")
print("\nSave Success!!!\n")
