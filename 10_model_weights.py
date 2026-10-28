import torchvision
from torchvision.models import VGG16_Weights
from torch import nn

# weights=None（随机初始化，不加载权重）
vgg16_without_weights = torchvision.models.vgg16(weights=None)

# weights=VGG16_Weights.IMAGENET1K_V1（加载官方预训练权重）
# vgg16_with_weights = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

print(vgg16_without_weights)

# 添加自定义层
# vgg16_without_weights.classifier.add_module('add_linear', nn.Linear(1000, 10))
# print(vgg16_without_weights)

# 替换一层
vgg16_without_weights.classifier[6] = nn.Linear(4096, 10)
print(vgg16_without_weights)

'''
模型与权重示例

1. 模型的结构定义在源码包里，无需下载
2. 权重需要从网络上下载（本地没有缓存的情况）
3. 修改模型结构
'''