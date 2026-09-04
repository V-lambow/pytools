# Preproceed

图像预处理工具集，主要包含基于 90% 裁剪的直方图均衡化处理。

## 功能

- `equalize90.py`：对图像指定 ROI 区域进行 90% 保留的直方图均衡化
  - 裁剪掉直方图两端各 5% 的极端像素值
  - 使用 LUT 映射增强对比度
  - 输出原图、ROI 区域、均衡化结果及直方图对比图

## 使用方法

修改 `equalize90.py` 中的配置参数后运行：

```python
SRC = r"train-ztd/xxx.jpg"   # 源图片路径
OUT_DIR = "./test"            # 输出目录
ROI = (0, 630, 1280, 1024)   # ROI 区域 (x, y, w, h)
KEEP = 0.90                   # 保留中间 90%
```

```bash
python equalize90.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `original.png` | 原始图片 |
| `roi.png` | 裁剪的 ROI 区域 |
| `roi_equalized.png` | 均衡化后的 ROI |
| `full_equalized.png` | 完整图片均衡化结果 |
| `region_marked.png` | 标注 ROI 区域的图片 |
| `comparison.png` | 原图与均衡化结果对比（含直方图） |

## 依赖

- opencv-python
- numpy
- matplotlib
