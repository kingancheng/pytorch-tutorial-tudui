import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1，模型结构+模型参数 全部打包存储到一个文件
# torch.save(vgg16, "vgg16_1.pth")

# 保存方式2，只保存模型参数（推荐）
torch.save(vgg16.state_dict(), "vgg16_2.pth")
