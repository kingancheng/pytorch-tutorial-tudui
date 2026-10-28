import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# 平均绝对误差
loss = nn.L1Loss()
result = loss(inputs, targets)

# 均方误差
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)

print(result)
print(result_mse)


x = torch.tensor([0.1, 0.2, 0.3]) # 模型输出的原始logits，表示在3个类别的打分
y = torch.tensor([1]) # 实际分类标签的索引
x = torch.reshape(x, (1, 3)) # 将x的形状从(3,)改为(1, 3)
loss_cross = nn.CrossEntropyLoss() # 交叉熵损失，用于分类
result_cross = loss_cross(x, y)
print(result_cross)

'''
几个损失函数示例

CrossEntropyLoss() 内部会计算Softmax，因此输入必须是logits。计算过程：
1. 将 logits 通过 Softmax 得到概率
2. 取真实类别的概率，例如上面的真实标签索引是1，所以对应第二个类别，即0.2对应的概率，
使用 loss = -log(p) 计算损失
'''