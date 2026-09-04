import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path


def load_json_annotations(json_path):
    """加载Labelme格式的JSON标注文件"""
    annotations = []
    if json_path and os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            if shape.get('shape_type') == 'rectangle':
                points = shape.get('points', [])
                if len(points) >= 2:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    annotations.append({
                        'label': shape.get('label', ''),
                        'width': width,
                        'height': height,
                        'points': points
                    })
    return annotations


def downsample_and_pad(img, scale=0.5):
    """降采样后pad回原尺寸"""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_h = h - new_h
    pad_w = w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded


def transform_json(json_path, output_path, scale=0.5):
    """变换JSON坐标"""
    if not json_path or not json_path.exists():
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        if points:
            new_points = []
            for p in points:
                new_x = int(p[0] * scale)
                new_y = int(p[1] * scale)
                new_points.append([new_x, new_y])
            shape['points'] = new_points
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_areas(annotations):
    """计算所有标注框的像素面积"""
    areas = []
    for ann in annotations:
        area = ann['width'] * ann['height']
        areas.append(area)
    return areas


def get_image_annotation_pairs(images_dir):
    """获取图片和对应的标注文件对"""
    pairs = []
    images_dir = Path(images_dir)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 检查是否有兄弟labels目录
    parent_dir = images_dir.parent
    labels_dir = parent_dir / "labels"
    has_labels_dir = images_dir.name.lower() == "images" and labels_dir.exists()
    
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in image_extensions:
            # 查找同名的 .json 文件（在同一目录）
            json_path = images_dir / (img_path.stem + ".json")
            
            # 查找同名的 .txt 文件（YOLO标注）
            if has_labels_dir:
                txt_path = labels_dir / (img_path.stem + ".txt")
            else:
                txt_path = images_dir / (img_path.stem + ".txt")
            
            pairs.append({
                'image': img_path,
                'json': json_path if json_path.exists() else None,
                'annotation': txt_path if txt_path.exists() else None
            })
    
    return pairs


def collect_all_areas(pairs):
    """收集所有标注框的面积"""
    all_areas = []
    valid_pairs = []
    
    for pair in pairs:
        img = cv2.imread(str(pair['image']))
        if img is None:
            continue
        
        if pair.get('json') is None or not pair['json'].exists():
            continue
        
        annotations = load_json_annotations(pair['json'])
        
        if annotations:
            areas = calculate_areas(annotations)
            all_areas.extend(areas)
            valid_pairs.append({
                'pair': pair,
                'areas': areas,
                'max_area': max(areas)
            })
    
    return all_areas, valid_pairs


def print_area_distribution(all_areas):
    """打印面积分布"""
    if not all_areas:
        print("没有找到有效的标注数据")
        return None
    
    areas = np.array(all_areas)
    
    print("\n===== 标注框面积分布（像素²） =====")
    print(f"总标注框数: {len(areas)}")
    print(f"最小面积: {areas.min():.0f}")
    print(f"最大面积: {areas.max():.0f}")
    print(f"平均面积: {areas.mean():.0f}")
    print(f"中位数面积: {np.median(areas):.0f}")
    
    # 自动生成分区
    min_val = areas.min()
    max_val = areas.max()
    
    # 使用对数刻度分区，更适合面积分布
    if min_val > 0 and max_val > 0:
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        num_bins = 10
        log_bins = np.linspace(log_min, log_max, num_bins + 1)
        bins = 10 ** log_bins
    else:
        bins = np.linspace(min_val, max_val, 11)
    
    print("\n===== 分区间统计（像素²） =====")
    for i in range(len(bins) - 1):
        count = np.sum((areas >= bins[i]) & (areas < bins[i+1]))
        percentage = count / len(areas) * 100
        bar = '█' * int(percentage / 2)
        print(f"[{bins[i]:.0f}, {bins[i+1]:.0f}): {count:4d} ({percentage:5.1f}%) {bar}")
    
    # 额外显示一些常用的面积阈值参考
    print("\n===== 常用面积参考 =====")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(areas, p)
        print(f"第{p}%分位数: {val:.0f} 像素²")
    
    return areas


def main():
    print("===== 图片面积筛选工具 =====")
    print("支持Labelme JSON格式标注文件")
    print("按标注框的绝对像素面积进行筛选")
    print("筛选后图片会降采样(0.5x)并pad回原尺寸，JSON坐标同步变换\n")
    
    images_dir = input("请输入图片目录路径: ").strip()
    if not os.path.isdir(images_dir):
        print(f"错误: 目录 '{images_dir}' 不存在")
        return
    
    print("\n正在扫描图片和标注文件...")
    pairs = get_image_annotation_pairs(images_dir)
    print(f"找到 {len(pairs)} 个图片文件")
    
    if not pairs:
        print("未找到任何图片文件")
        return
    
    print("\n正在计算面积...")
    all_areas, valid_pairs = collect_all_areas(pairs)
    areas = print_area_distribution(all_areas)
    
    if areas is None or len(areas) == 0:
        print("没有可处理的数据")
        return
    
    print(f"\n请输入筛选范围，格式如: {areas.min():.0f}~{areas.max():.0f}")
    range_input = input("面积范围（像素²）: ").strip()
    
    if '~' not in range_input:
        print("错误: 格式应为 最小值~最大值，例如 1000~5000")
        return
    
    try:
        parts = range_input.split('~')
        min_area = float(parts[0].strip())
        max_area = float(parts[1].strip())
    except ValueError:
        print("错误: 输入格式不正确")
        return
    
    if min_area > max_area:
        min_area, max_area = max_area, min_area
        print(f"已自动调整范围为: {min_area:.0f} ~ {max_area:.0f}")
    
    # 筛选：只要图片中最大面积的标注框在范围内就算通过
    print("\n正在筛选图片...")
    filtered = [vp for vp in valid_pairs if min_area <= vp['max_area'] <= max_area]
    print(f"筛选后剩余 {len(filtered)} 张图片")
    
    if not filtered:
        print("没有符合条件的图片")
        return
    
    output_dir = input("\n请输入输出目录路径: ").strip()
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    
    print("\n正在处理文件（降采样+坐标变换）...")
    scale = 0.5
    for i, vp in enumerate(filtered):
        # 处理图片：降采样+pad
        img = cv2.imread(str(vp['pair']['image']))
        if img is not None:
            img_processed = downsample_and_pad(img, scale)
            img_dst = output_dir / "images" / vp['pair']['image'].name
            cv2.imwrite(str(img_dst), img_processed)
        
        # 处理JSON：坐标变换
        if vp['pair'].get('json') and vp['pair']['json'].exists():
            json_dst = output_dir / "images" / vp['pair']['json'].name
            transform_json(vp['pair']['json'], json_dst, scale)
        
        # 复制txt标注文件（不变换）
        if vp['pair']['annotation'] and vp['pair']['annotation'].exists():
            ann_dst = output_dir / "labels" / vp['pair']['annotation'].name
            shutil.copy2(vp['pair']['annotation'], ann_dst)
        
        if (i + 1) % 100 == 0 or (i + 1) == len(filtered):
            print(f"  已处理 {i + 1}/{len(filtered)}")
    
    print(f"\n完成! 共复制 {len(filtered)} 张图片到: {output_dir}")


if __name__ == "__main__":
    main()
