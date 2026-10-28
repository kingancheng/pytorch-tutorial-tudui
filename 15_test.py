import torch
import torchvision
from PIL import Image
from torch import nn
from model import *

image_path = "imgs/test.jpg"
image = Image.open(image_path).convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape) # torch.Size([3, 32, 32])
image = torch.reshape(image, (-1, 3, 32, 32)) # 转换为 (batch_size, 3, 32, 32)
print(image.shape) # torch.Size([1, 3, 32, 32])

model = Tudui()
model.load_state_dict(torch.load("model_save/tudui_9.pth", map_location=torch.device('cpu')))

model.eval()
with torch.no_grad():
    output = model(image)


print(output) # 所有分类的原始logits
print(output.argmax(1)) # 最高分类别的索引

'''
使用训练好的模型进行预测
'''
