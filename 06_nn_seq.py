import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential


# Sequential按顺序逐层调用，无需forward中手动调用每层
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


module = MyModule()
print(module)
input = torch.ones((64, 3, 32, 32))
output = module(input)
print(output.shape)

'''
Sequential的使用
'''
