import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid, Linear


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
        self.linear1 = Linear(64, 10)

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


module = MyModule()


'''
常用线性层、非线性层示例
'''
