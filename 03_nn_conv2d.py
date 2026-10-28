import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


module = MyModule()

writer = SummaryWriter("logs/03")

step = 0
for data in dataloader:
    imgs, targets = data
    output = module(imgs)
    print(imgs.shape) # torch.Size([64, 3, 32, 32])
    print(output.shape) # torch.Size([64, 6, 30, 30])

    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])  -> [xxx, 3, 30, 30]
    # 注意这个转换只是将6通道转换为3通道，用于可视化，并不一定具备实际意义
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)

    step = step + 1

writer.close()

'''
卷积层使用基础示例
读取数据集 → 卷积层计算 → 输入输出记录与可视化
'''