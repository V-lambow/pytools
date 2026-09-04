# Label Statistics

缺陷标签统计可视化工具，用于分析 JSON 格式标注数据中各类缺陷的数量和面积分布。

## 功能

- 扫描指定目录中的 `.json` 标注文件
- 统计每个缺陷类别的出现次数
- 计算矩形标注框的像素面积
- 生成缺陷数量柱状图和面积分布箱线图

## 使用方法

```bash
python label_statistics.py
```

运行后按提示输入：

1. 标注文件目录路径（包含 `.json` 文件）
2. 输出目录路径

## 输出文件

| 文件 | 说明 |
|------|------|
| `defect_count_statistics.png` | 各类别缺陷数量柱状图 |
| `defect_area_statistics.png` | 各类别缺陷面积分布箱线图 |

## 标注格式

支持 LabelMe 风格的 JSON 标注：

```json
{
  "shapes": [
    {
      "label": "缺陷类别名",
      "shape_type": "rectangle",
      "points": [[x1, y1], [x2, y2]]
    }
  ]
}
```

## 依赖

- matplotlib
- numpy
