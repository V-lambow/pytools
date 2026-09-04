# Split Dataset

按缺陷类别智能分割数据集的工具，将图片和标注按比例随机分配到 train/val。

## 功能

- 统计每种缺陷类别的图片数量
- 少于 20 张的类别全部归入 train，超过 20 张的按比例分配
- 自动复制图片及对应的 `.json` / `.txt` / `.xml` 标注文件
- 生成 `classes.txt` 类别文件
- 输出各类别分配汇总表

## 使用方法

```bash
python split_dataset.py
```

按提示输入：

1. 图片文件夹路径（图片与标注文件同目录）
2. train:val 比例（如 `8:2` 或 `0.8`）

## 输出结构

```
image_dir/
├── train/images/    # 训练集
├── val/images/      # 验证集
└── classes.txt      # 类别列表
```

## 标注格式

支持以下格式的标注文件（与图片同名）：

- `.json`（LabelMe 格式）
- `.txt`（YOLO 格式）
- `.xml`（Pascal VOC 格式）

## 依赖

- 标准库即可（os, shutil, random, json）
