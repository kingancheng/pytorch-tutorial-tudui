import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
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
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


module = MyModule()

writer = SummaryWriter("logs/04")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = module(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()


'''
max池化层基础示例
'''