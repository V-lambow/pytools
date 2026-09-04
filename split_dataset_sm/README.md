# Split Dataset SM

基于 **HBB 框大小** KMeans 聚类的数据集分割工具（k=2，分为 small/large 两组）。

## 功能

1. 统计每个标签的平均 HBB 框面积
2. 对面积做 KMeans 聚类，分为 s 组（小框）和 m 组（大框）
3. 每张图取出现次数最少的标签为主标签进行分配
4. 跨组图片同时进 s 和 m 组，各组只保留对应标签
5. 自动生成 `classes.txt` 和 YOLO 格式 `yaml` 配置文件

## 使用方法

```bash
python split_s_m.py
```

按提示输入：

1. 图片文件夹路径
2. train:val 比例（如 `8:2` 或 `0.8`）

## 输出结构

```
image_dir/
├── s/
│   ├── train/images/
│   ├── val/images/
│   └── classes.txt
├── m/
│   ├── train/images/
│   ├── val/images/
│   └── classes.txt
├── dataset_name_s.yaml
└── dataset_name_m.yaml
```

## 依赖

- scikit-learn
- numpy
