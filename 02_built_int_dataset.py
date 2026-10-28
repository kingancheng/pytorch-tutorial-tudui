import torchvision
from torch.utils.tensorboard import SummaryWriter


# 加载 CIFAR10 数据集，不使用 transform 转换
# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)


'''打印数据集的信息'''
# print(test_set[0]) # 每个元素是一个元组，(img, target)
# img, target = test_set[0]
# print(img) # 图像数据
# print(target) # 标签对应的索引

# print(test_set.classes) # 标签列表
# print(test_set.classes[target])
# img.show()

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 加载数据集，并使用 transform 转换
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)


writer = SummaryWriter("logs/02")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()


'''
要点：使用built-in dataset
文档：https://docs.pytorch.org/vision/stable/datasets.html#built-in-datasets

CIFAR-10 数据集下载：https://cave.cs.toronto.edu/kriz/cifar.html
手动下载后将数据集压缩包放在 root 参数指定的路径下，运行代码会自动解压
'''
