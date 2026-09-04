# Split Dataset Freq

基于**类别出现频次** KMeans 聚类的数据集分割工具（k=2，分为 high/low 两组）。

## 功能

1. 统计每个标签在多少张图片中出现（出现频次）
2. 对频次做对数变换后 KMeans 聚类，分为高频组和低频组
3. 每张图取出现次数最少的标签为主标签进行分配
4. 跨组图片同时进 high 和 low 组，各组只保留对应标签
5. 自动生成 `classes.txt` 和 YOLO 格式 `yaml` 配置文件

## 使用方法

```bash
python split_freq.py
```

按提示输入：

1. 图片文件夹路径
2. train:val 比例（如 `8:2` 或 `0.8`）

## 输出结构

```
image_dir/
├── high/
│   ├── train/images/
│   ├── val/images/
│   └── classes.txt
├── low/
│   ├── train/images/
│   ├── val/images/
│   └── classes.txt
├── dataset_name_high.yaml
└── dataset_name_low.yaml
```

## 依赖

- scikit-learn
- numpy
