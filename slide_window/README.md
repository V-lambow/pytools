# Slide Window

滑动窗口图像切割工具，支持 YOLO/LabelMe 标注的自动裁剪与分发。

## 版本说明

| 版本 | 文件 | 说明 |
|------|------|------|
| 基础版 | `slide_window.py` | 交互式输入，逐图串行处理 |
| 优化版 | `slide_window_optimized.py` | 多进程并行处理，性能更优 |
| 单图版 | `slide_window_single.py` | 只处理单张图片，智能过滤小框 |
| 基准测试 | `benchmark.py` | 对比各版本性能 |

## 功能

- 将大图按指定行数和列数切割为子块
- 支持设置子块间的交叠像素
- 支持整图边框扩展/缩减（+ 向外扩充，- 向内缩减）
- 自动裁剪并保存对应标注框到子块中
- 过滤被切割边界截断的过小标注框

## 使用方法

```bash
python slide_window_optimized.py
```

按提示输入：

1. 数据集目录路径（图片与 `.json` 标注同目录）
2. 行数,列数（如 `3,4`）
3. 交叠像素（如 `32`）
4. 边框扩展像素（`+100` / `-50` / `0`）
5. 输出目录路径
6. 并行处理数（默认 4）

## 输出结构

```
output/
├── image_r0_c0.png
├── image_r0_c0.json
├── image_r0_c1.png
├── image_r0_c1.json
└── ...
```

## 依赖

- Pillow
- numpy
