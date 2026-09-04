"""
数据集按类别出现频次聚类分割工具

功能：
1. 遍历所有标注，统计每个标签出现的图片数（出现频次）
2. 每张图取全局出现次数最少的标签为主标签
3. 对标签的出现频次做k=2聚类，分为high组(高频)和low组(低频)
4. 按输入比例随机拆分train/val（主标签为准，<20张全进train）
5. train和val中，根据图片所有标签的聚类归属分high/low：
   - 全部标签在high → 只进high组
   - 全部标签在low → 只进low组
   - 跨high和low → 同时进high和low，各组只保留对应聚类的标签
6. 输出到输入文件夹内，生成classes.txt和yaml配置文件

输出结构：
  high/train/images/
  high/val/images/
  low/train/images/
  low/val/images/
"""

import os
import shutil
import random
import json
import numpy as np
from sklearn.cluster import KMeans


def get_image_extensions():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def load_json_labels(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = set()
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if label:
                labels.add(label)
        return labels
    except Exception:
        return set()


def analyze_dataset(image_dir):
    image_extensions = get_image_extensions()
    all_files = os.listdir(image_dir)

    images = []
    for file in all_files:
        full_path = os.path.join(image_dir, file)
        if os.path.isdir(full_path):
            continue
        ext = os.path.splitext(file)[1].lower()
        if ext in image_extensions:
            images.append(file)

    label_image_count = {}
    image_labels = {}
    image_primary_label = {}

    for image_name in images:
        json_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.json')
        if not os.path.exists(json_path):
            continue
        labels = load_json_labels(json_path)
        if not labels:
            continue

        image_labels[image_name] = labels

        for label in labels:
            label_image_count[label] = label_image_count.get(label, 0) + 1

    for image_name, labels in image_labels.items():
        rarest = min(labels, key=lambda l: label_image_count.get(l, 0))
        image_primary_label[image_name] = rarest

    classified_images = {}
    for image_name, label in image_primary_label.items():
        classified_images.setdefault(label, []).append(image_name)

    return (images, label_image_count, image_labels,
            image_primary_label, classified_images)


def cluster_labels_by_frequency(label_image_count):
    labels = list(label_image_count.keys())
    freq = np.array([label_image_count[l] for l in labels]).reshape(-1, 1)
    log_freq = np.log1p(freq)

    if len(labels) < 2:
        return set(labels), set()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(log_freq)

    cluster_centers = kmeans.cluster_centers_.flatten()
    high_cluster = 0 if cluster_centers[0] > cluster_centers[1] else 1

    high_labels = set()
    low_labels = set()
    for label, cluster_id in zip(labels, kmeans.labels_):
        if cluster_id == high_cluster:
            high_labels.add(label)
        else:
            low_labels.add(label)

    return high_labels, low_labels


def split_by_class(classified_images, train_ratio, min_threshold=20):
    train_images = []
    val_images = []

    for label, image_list in classified_images.items():
        random.shuffle(image_list)
        count = len(image_list)
        if count < min_threshold:
            if count == 1:
                train_images.extend(image_list)
                print(f"  {label}: {count} 张 -> 全部归入 train")
            else:
                train_images.extend(image_list[:count - 1])
                val_images.append(image_list[count - 1])
                print(f"  {label}: {count} 张 (< {min_threshold}) -> train {count-1} 张, val 1 张")
        else:
            split_idx = int(count * train_ratio)
            train_images.extend(image_list[:split_idx])
            val_images.extend(image_list[split_idx:])

    return train_images, val_images


def copy_image(src_dir, image_name, dest_dir):
    image_src = os.path.join(src_dir, image_name)
    if os.path.exists(image_src):
        shutil.copy2(image_src, dest_dir)


def copy_with_trimmed_annotation(src_dir, image_name, dest_dir, allowed_labels):
    copy_image(src_dir, image_name, dest_dir)

    json_src = os.path.join(src_dir, os.path.splitext(image_name)[0] + '.json')
    if not os.path.exists(json_src):
        return

    with open(json_src, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['shapes'] = [
        shape for shape in data.get('shapes', [])
        if shape.get('label', '') in allowed_labels
    ]

    json_dst = os.path.join(dest_dir, os.path.splitext(image_name)[0] + '.json')
    with open(json_dst, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_classes_txt(path, labels):
    with open(path, 'w', encoding='utf-8') as f:
        for label in sorted(labels):
            f.write(label + '\n')


def write_yaml(path, dataset_root, train_rel, val_rel, labels):
    sorted_labels = sorted(labels)
    names_dict = {i: label for i, label in enumerate(sorted_labels)}
    lines = [
        f"# Ultralytics AGPL-3.0 License",
        f"#",
        f"# Train/val sets",
        f"path: {dataset_root}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"test: ",
        f"",
        f"# Classes",
        f"names:",
    ]
    for i, label in enumerate(sorted_labels):
        lines.append(f"  {i}: {label}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    image_dir = input("请输入图片文件夹路径: ").strip()
    if not os.path.exists(image_dir):
        print(f"错误: 文件夹 '{image_dir}' 不存在")
        return

    ratio_str = input("请输入 train:val 比例 (例如 8:2 或 0.8): ").strip()
    if ':' in ratio_str:
        parts = ratio_str.split(':')
        train_ratio = int(parts[0]) / (int(parts[0]) + int(parts[1]))
    else:
        train_ratio = float(ratio_str)

    val_ratio = 1 - train_ratio
    print(f"\ntrain 比例: {train_ratio:.0%}, val 比例: {val_ratio:.0%}")

    print("\n正在分析数据集...")
    (images, label_image_count, image_labels,
     image_primary_label, classified_images) = analyze_dataset(image_dir)

    if not images:
        print(f"错误: 在 '{image_dir}' 中未找到图片文件")
        return

    print(f"\n共找到 {len(images)} 张图片，{len(label_image_count)} 种标签:")
    print("-" * 50)
    print(f"  {'标签':<15} {'出现图片数':>10}")
    print("-" * 50)
    for label, count in sorted(label_image_count.items(), key=lambda x: x[1]):
        print(f"  {label:<15} {count:>10}")
    print("-" * 50)

    print("\n正在进行出现频次k=2聚类...")
    high_labels, low_labels = cluster_labels_by_frequency(label_image_count)

    print(f"\n  high组标签 (高频, {len(high_labels)}个):")
    for l in sorted(high_labels, key=lambda x: label_image_count[x], reverse=True):
        print(f"    {l}: 出现 {label_image_count[l]} 张")

    print(f"\n  low组标签 (低频, {len(low_labels)}个):")
    for l in sorted(low_labels, key=lambda x: label_image_count[x], reverse=True):
        print(f"    {l}: 出现 {label_image_count[l]} 张")

    print(f"\n按标签类别分配 train/val:")
    print("-" * 50)
    train_images, val_images = split_by_class(classified_images, train_ratio)
    print("-" * 50)

    high_train_dir = os.path.join(image_dir, 'high', 'train', 'images')
    high_val_dir = os.path.join(image_dir, 'high', 'val', 'images')
    low_train_dir = os.path.join(image_dir, 'low', 'train', 'images')
    low_val_dir = os.path.join(image_dir, 'low', 'val', 'images')
    for d in [high_train_dir, high_val_dir, low_train_dir, low_val_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n正在分配训练集到high/low组...")
    train_high_count = 0
    train_low_count = 0
    for image_name in train_images:
        labels = image_labels.get(image_name, set())
        has_high = bool(labels & high_labels)
        has_low = bool(labels & low_labels)

        if has_high:
            copy_with_trimmed_annotation(image_dir, image_name, high_train_dir, high_labels)
            train_high_count += 1
        if has_low:
            copy_with_trimmed_annotation(image_dir, image_name, low_train_dir, low_labels)
            train_low_count += 1

    print(f"  训练集 high组: {train_high_count} 张, low组: {train_low_count} 张")

    print(f"\n正在分配验证集到high/low组...")
    val_high_count = 0
    val_low_count = 0
    for image_name in val_images:
        labels = image_labels.get(image_name, set())
        has_high = bool(labels & high_labels)
        has_low = bool(labels & low_labels)

        if has_high:
            copy_with_trimmed_annotation(image_dir, image_name, high_val_dir, high_labels)
            val_high_count += 1
        if has_low:
            copy_with_trimmed_annotation(image_dir, image_name, low_val_dir, low_labels)
            val_low_count += 1

    print(f"  验证集 high组: {val_high_count} 张, low组: {val_low_count} 张")

    folder_name = os.path.basename(os.path.normpath(image_dir))

    high_classes_path = os.path.join(image_dir, 'high', 'classes.txt')
    write_classes_txt(high_classes_path, high_labels)

    low_classes_path = os.path.join(image_dir, 'low', 'classes.txt')
    write_classes_txt(low_classes_path, low_labels)

    high_yaml_path = os.path.join(image_dir, f'{folder_name}_high.yaml')
    write_yaml(high_yaml_path, image_dir, 'high/train/images', 'high/val/images', high_labels)

    low_yaml_path = os.path.join(image_dir, f'{folder_name}_low.yaml')
    write_yaml(low_yaml_path, image_dir, 'low/train/images', 'low/val/images', low_labels)

    print(f"\n{'='*60}")
    print(f"  分割结果汇总")
    print(f"{'='*60}")
    print(f"  总图片数: {len(images)}")
    print(f"  train原始: {len(train_images)}, val原始: {len(val_images)}")
    print(f"  high组 train: {train_high_count}, val: {val_high_count}")
    print(f"  low组 train: {train_low_count}, val: {val_low_count}")
    print(f"{'='*60}")
    print(f"\n输出目录: {image_dir}")
    print(f"  high/train/images/     high/classes.txt")
    print(f"  high/val/images/")
    print(f"  low/train/images/      low/classes.txt")
    print(f"  low/val/images/")
    print(f"  {folder_name}_high.yaml ({len(high_labels)} 个标签)")
    print(f"  {folder_name}_low.yaml ({len(low_labels)} 个标签)")
    print(f"\n完成!")


if __name__ == "__main__":
    main()