# -*- coding: utf-8 -*-
"""一次性迁移脚本：将 ants_label / bees_label 目录下每图一个 .txt 的标签
整合为 dataset/train/labels.json（key 为相对 root_dir 的图片路径，value 为标签）。
运行后即可删除旧的 label 目录。"""
import json
import os


def migrate(root_dir):
    labels = {}
    for label_dir in ("ants_label", "bees_label"):
        label_path = os.path.join(root_dir, label_dir)
        for name in sorted(os.listdir(label_path)):
            with open(os.path.join(label_path, name), 'r') as f:
                label = f.readline().strip()
            # 标签文件名与图片文件名相同（仅扩展名不同），图片存放在同名类别目录下
            image_name = os.path.splitext(name)[0]
            # ants_label -> ants, bees_label -> bees
            image_dir = label_dir.replace('_label', '')
            found = None
            # 先按“图片名 + 常见扩展名”匹配，再兼容标签文件名本身已是完整图片名的情况
            # （如 formica.jpeg.txt 对应 formica.jpeg.jpg，splitext 后只剩 formica.jpeg）
            candidates = [image_name + ext for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.gif')] + [image_name]
            for cand in candidates:
                if os.path.exists(os.path.join(root_dir, image_dir, cand)):
                    found = cand
                    break
            assert found, f"未找到对应图片: {name} -> {image_dir}/{image_name}.*"
            # 统一使用正斜杠，跨平台读取更稳定
            key = f"{image_dir}/{found}"
            labels[key] = label
    out_path = os.path.join(root_dir, "labels.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    print(f"共整合 {len(labels)} 条标签 -> {out_path}")


if __name__ == '__main__':
    migrate("dataset/train")
