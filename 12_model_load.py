import torch
import torchvision
from torch import nn


# 加载方式1，同时加载模型结构+模型参数
# model1 = torch.load("vgg16_1.pth")
# print(model1)

# 加载方式2，只加载模型参数，模型结构单独建立
state_dict = torch.load("vgg16_2.pth")
model2 = torchvision.models.vgg16(weights=None)
ret = model2.load_state_dict(state_dict) # 返回值用于验证是否加载成功
print(ret)

# map_location="cpu" 强制将权重映射到 CPU 内存加载
# torch.load 默认会把张量还原到保存时所在的设备（如 cuda:0），
# 若训练时用 GPU 保存、而当前机器没有 CUDA，不加该参数会直接报错：
# RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False
torch.load("vgg16_2.pth", map_location="cpu")