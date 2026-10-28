import os
from datetime import datetime
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from torch import nn
from torch.utils.data import DataLoader

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset",
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset",
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练集长度：{train_data_size}")
print(f"测试集长度：{test_data_size}")


# DataLoader 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# Tensorboard 记录
Timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"logs/train-{Timestamp}")

for i in range(epoch):
    print(f"-------第 {i+1} 轮训练开始-------")

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()   # 清空上一步残留的梯度
        loss.backward()         # 反向传播，计算当前每个参数的梯度
        optimizer.step()        # 用这些梯度更新模型参数

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print(f"训练次数：{total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    total_test_step += 1

    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的正确率: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    os.makedirs("model_save", exist_ok=True)
    torch.save(tudui.state_dict(), f"model_save/tudui_{i}.pth")
    print("模型已保存")

writer.close()

'''
一个完整的PyTorch训练流程，CPU版本
'''