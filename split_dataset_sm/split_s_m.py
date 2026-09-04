"""
数据集按HBB框大小聚类分割工具

功能：
1. 遍历所有标注，统计标签出现次数，计算每个标签的平均HBB面积
2. 每张图取全局出现次数最少的标签为主标签
3. 对标签的平均HBB面积做k=2聚类，分为s组(小框)和m组(大框)
4. 按输入比例随机拆分train/val（主标签为准，<20张全进train）
5. train和val中，根据图片所有标签的聚类归属分s/m：
   - 全部标签在s → 只进s组
   - 全部标签在m → 只进m组
   - 跨s和m → 同时进s和m，各组只保留对应聚类的标签
6. 输出到输入文件夹内，生成classes.txt和yaml配置文件

输出结构：
  s/train/images/
  s/val/images/
  m/train/images/
  m/val/images/
"""

import os
import shutil
import random
import json
import numpy as np
from sklearn.cluster import KMeans


def get_image_extensions():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def load_json_labels_and_boxes(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = set()
        boxes = {}
        for shape in data.get('shapes', []):
            label = shape.get('label', '')
            if not label:
                continue
            labels.add(label)
            if shape.get('shape_type') == 'rectangle':
                pts = shape.get('points', [])
                if len(pts) == 2:
                    w = abs(pts[1][0] - pts[0][0])
                    h = abs(pts[1][1] - pts[0][1])
                    area = w * h
                    boxes.setdefault(label, []).append(area)
        return labels, boxes
    except Exception:
        return set(), {}


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
    label_total_area = {}
    label_instance_count = {}
    image_labels = {}
    image_primary_label = {}

    for image_name in images:
        json_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.json')
        if not os.path.exists(json_path):
            continue
        labels, boxes = load_json_labels_and_boxes(json_path)
        if not labels:
            continue

        image_labels[image_name] = labels

        for label in labels:
            label_image_count[label] = label_image_count.get(label, 0) + 1
            if label in boxes:
                label_total_area[label] = label_total_area.get(label, 0) + sum(boxes[label])
                label_instance_count[label] = label_instance_count.get(label, 0) + len(boxes[label])

    for image_name, labels in image_labels.items():
        rarest = min(labels, key=lambda l: label_image_count.get(l, 0))
        image_primary_label[image_name] = rarest

    label_mean_area = {}
    for label in label_total_area:
        if label_instance_count.get(label, 0) > 0:
            label_mean_area[label] = label_total_area[label] / label_instance_count[label]

    classified_images = {}
    for image_name, label in image_primary_label.items():
        classified_images.setdefault(label, []).append(image_name)

    return (images, label_image_count, label_mean_area, image_labels,
            image_primary_label, classified_images)


def cluster_labels_by_area(label_mean_area):
    labels = list(label_mean_area.keys())
    areas = np.array([label_mean_area[l] for l in labels]).reshape(-1, 1)

    if len(labels) < 2:
        return set(labels), set()

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(areas)

    cluster_centers = kmeans.cluster_centers_.flatten()
    small_cluster = 0 if cluster_centers[0] < cluster_centers[1] else 1

    s_labels = set()
    m_labels = set()
    for label, cluster_id in zip(labels, kmeans.labels_):
        if cluster_id == small_cluster:
            s_labels.add(label)
        else:
            m_labels.add(label)

    return s_labels, m_labels


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
    (images, label_image_count, label_mean_area, image_labels,
     image_primary_label, classified_images) = analyze_dataset(image_dir)

    if not images:
        print(f"错误: 在 '{image_dir}' 中未找到图片文件")
        return

    print(f"\n共找到 {len(images)} 张图片，{len(label_image_count)} 种标签:")
    print("-" * 60)
    print(f"  {'标签':<15} {'数量':>6} {'平均HBB面积':>15}")
    print("-" * 60)
    for label, count in sorted(label_image_count.items(), key=lambda x: x[1]):
        mean_area = label_mean_area.get(label, 0)
        print(f"  {label:<15} {count:>6} {mean_area:>15.1f}")
    print("-" * 60)

    print("\n正在进行HBB面积k=2聚类...")
    s_labels, m_labels = cluster_labels_by_area(label_mean_area)

    print(f"\n  s组标签 (小框, {len(s_labels)}个):")
    for l in sorted(s_labels):
        print(f"    {l}: 平均面积 {label_mean_area.get(l, 0):.1f}")

    print(f"\n  m组标签 (大框, {len(m_labels)}个):")
    for l in sorted(m_labels):
        print(f"    {l}: 平均面积 {label_mean_area.get(l, 0):.1f}")

    print(f"\n按标签类别分配 train/val:")
    print("-" * 60)
    train_images, val_images = split_by_class(classified_images, train_ratio)
    print("-" * 60)

    s_train_dir = os.path.join(image_dir, 's', 'train', 'images')
    s_val_dir = os.path.join(image_dir, 's', 'val', 'images')
    m_train_dir = os.path.join(image_dir, 'm', 'train', 'images')
    m_val_dir = os.path.join(image_dir, 'm', 'val', 'images')
    for d in [s_train_dir, s_val_dir, m_train_dir, m_val_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n正在分配训练集到s/m组...")
    train_s_count = 0
    train_m_count = 0
    for image_name in train_images:
        labels = image_labels.get(image_name, set())
        has_s = bool(labels & s_labels)
        has_m = bool(labels & m_labels)

        if has_s:
            copy_with_trimmed_annotation(image_dir, image_name, s_train_dir, s_labels)
            train_s_count += 1
        if has_m:
            copy_with_trimmed_annotation(image_dir, image_name, m_train_dir, m_labels)
            train_m_count += 1

    print(f"  训练集 s组: {train_s_count} 张, m组: {train_m_count} 张")

    print(f"\n正在分配验证集到s/m组...")
    val_s_count = 0
    val_m_count = 0
    for image_name in val_images:
        labels = image_labels.get(image_name, set())
        has_s = bool(labels & s_labels)
        has_m = bool(labels & m_labels)

        if has_s:
            copy_with_trimmed_annotation(image_dir, image_name, s_val_dir, s_labels)
            val_s_count += 1
        if has_m:
            copy_with_trimmed_annotation(image_dir, image_name, m_val_dir, m_labels)
            val_m_count += 1

    print(f"  验证集 s组: {val_s_count} 张, m组: {val_m_count} 张")

    folder_name = os.path.basename(os.path.normpath(image_dir))

    s_classes_path = os.path.join(image_dir, 's', 'classes.txt')
    write_classes_txt(s_classes_path, s_labels)

    m_classes_path = os.path.join(image_dir, 'm', 'classes.txt')
    write_classes_txt(m_classes_path, m_labels)

    s_yaml_path = os.path.join(image_dir, f'{folder_name}_s.yaml')
    write_yaml(s_yaml_path, image_dir, 's/train/images', 's/val/images', s_labels)

    m_yaml_path = os.path.join(image_dir, f'{folder_name}_m.yaml')
    write_yaml(m_yaml_path, image_dir, 'm/train/images', 'm/val/images', m_labels)

    print(f"\n{'='*60}")
    print(f"  分割结果汇总")
    print(f"{'='*60}")
    print(f"  总图片数: {len(images)}")
    print(f"  train原始: {len(train_images)}, val原始: {len(val_images)}")
    print(f"  s组 train: {train_s_count}, val: {val_s_count}")
    print(f"  m组 train: {train_m_count}, val: {val_m_count}")
    print(f"{'='*60}")
    print(f"\n输出目录: {image_dir}")
    print(f"  s/train/images/     s/classes.txt")
    print(f"  s/val/images/")
    print(f"  m/train/images/     m/classes.txt")
    print(f"  m/val/images/")
    print(f"  {folder_name}_s.yaml ({len(s_labels)} 个标签)")
    print(f"  {folder_name}_m.yaml ({len(m_labels)} 个标签)")
    print(f"\n完成!")


if __name__ == "__main__":
    main()
