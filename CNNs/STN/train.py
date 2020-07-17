import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import time
from models import STN
from config import cfg_net

data_path = cfg_net['data_path']
batch_size = 2
num_workers = 1
cuda = False
max_epochs = 100
base_lr = 0.1
adj_lr = 20
DEVICE = 'cuda:1'
device = torch.device(DEVICE)


def adjust_learning_rate(optimizer, step, decay_rate=0.5):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

class Gray2RGB(object):
    def __init__(self):
        super(Gray2RGB, self).__init__()

    def __call__(self, img):
        img = Image.Image.convert(img, "RGB")
        return img

train_transform = transforms.Compose([
    Gray2RGB(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
val_transform = transforms.Compose([
    Gray2RGB(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

train_set = dsets.MNIST(root=data_path, train=True, download=False, transform=train_transform)
val_set = dsets.MNIST(root=data_path, train=False, download=False, transform=val_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=cuda)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, pin_memory=cuda)

net = STN(in_channel=3, inshape=26)

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

print(' ****** start train .....')
for epoch in range(max_epochs):
    for step, sample in enumerate(train_loader):
        s = time.time()

        images, labels = sample
        out = net(images)

        optimizer.zero_grad()
        # a = sample['a'].to(device)
        # b = sample['b'].to(device)
        out = net(a)
        loss = F.l1_loss(b, out)
        # loss = dlibloss(sample, landmloss, net, device)
        # loss = loss_func(out, b)
        loss.backward()
        optimizer.step()
        print('Train step: {}/{} {}/{} Loss: {:.6f}  lr: {}  time: {:.3f}'.format(
            step + epoch * batch_size, max_epochs * batch_size, epoch, max_epochs, loss.item(),
            optimizer.param_groups[0]['lr'],
            time.time() - s
        ))
    if epoch % adj_lr == 0 and epoch != 0:
        adjust_learning_rate(optimizer, epoch)

torch.save(net.state_dict(), 'stn.pth')


