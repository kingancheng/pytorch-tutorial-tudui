import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

writer = SummaryWriter("logs/01")

class MyData(Dataset):

    def __init__(self, root_dir, image_dir, label_file, transform):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.label_file = label_file
        self.image_path = os.path.join(self.root_dir, self.image_dir)
        self.transform = transform
        # 标签不再每图一个文件，而是统一存放在一个 JSON 文件中：
        # key 为相对 root_dir 的图片路径（如 "ants/0013035.jpg"），value 为标签
        label_path = os.path.join(self.root_dir, self.label_file)
        with open(label_path, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.image_list = sorted(
            f"{self.image_dir}/{name}" for name in os.listdir(self.image_path))
        # 校验 JSON 中的 key 与图片目录一一对应
        assert set(self.image_list) <= set(self.labels.keys()), \
            "labels.json 中缺少部分图片的标签"

    def __getitem__(self, idx):
        img_key = self.image_list[idx]
        img_item_path = os.path.join(self.root_dir, img_key)
        label = self.labels[img_key]
        img = Image.open(img_item_path).convert('RGB')

        img = self.transform(img)
        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = "dataset/train"
    # ants 和 bees 共用一个标签文件 labels.json
    ants_dataset = MyData(root_dir, "ants", "labels.json", transform)
    bees_dataset = MyData(root_dir, "bees", "labels.json", transform)
    train_dataset = ants_dataset + bees_dataset

    # 打印第一个样本的图片形状
    print(train_dataset[0]['img'].shape) # torch.Size([3, 256, 256])，对应3个通道的张量

    # 创建 DataLoader（num_workers加载线程数）
    dataloader = DataLoader(train_dataset, batch_size=4, num_workers=2)

    # 添加一个样本图片到 TensorBoard
    writer.add_image('sample', train_dataset[0]['img'])
    
    for i, j in enumerate(dataloader):
        print(type(j)) # <class 'dict'>，字典由 __getitem__ 方法的返回值决定
        print(i, j['img'].shape) # torch.Size([batch_size, 3, 256, 256])，即每次对dataloader的遍历取得的是一批batch_size的样本
        writer.add_image("train_data_batch", make_grid(j['img']), i)
    
    writer.close()


''' 
要点：
1. 自定义数据集，继承 Dataset 类，并实现 __init__、__getitem__ 和 __len__ 等方法
2. PIL 读取图片
3. transforms 进行图像变换（缩放、转换为张量、归一化 等）
4. DataLoader 的创建与遍历
5. 使用 tensorboard 的 SummaryWriter 记录数据，并使用 tensorboard 进行可视化
6. 使用 make_grid 函数将多个图像拼接成一个图像

启动 TensorBoard 可视化界面命令：tensorboard --logdir=logs
'''