# Split Dataset Freq Area

基于**出现频次 + HBB 框面积**二维 KMeans 聚类的数据集分割工具。

## 功能

1. 统计每个标签的出现图片数（频次）和平均 HBB 面积
2. 构建二维特征 `[log1p(频次), log1p(平均面积)]`，使用 KMeans 聚类
3. 通过轮廓系数自动推荐最优聚类数 K，支持用户自定义
4. 按面积优先策略分配 train/val（极端面积优先进训练集）
5. 跨组图片同时进多个组，各组只保留对应聚类的标签
6. 自动生成 `classes.txt` 和 YOLO 格式 `yaml` 配置文件

## 使用方法

```bash
python split_freq_area.py
```

按提示输入：

1. 图片文件夹路径
2. train:val 比例（如 `8:2` 或 `0.8`）
3. 确认推荐的 K 值或输入自定义聚类数

## 输出结构

```
image_dir/
├── g0/
│   ├── train/images/
│   ├── val/images/
│   └── classes.txt
├── g1/
│   ├── train/images/
│   ├── val/images/
│   └── classes.txt
├── dataset_name_g0.yaml
└── dataset_name_g1.yaml
```

## 依赖

- scikit-learn
- numpy
