"""
图片和标注文件按缺陷类别智能分割工具
将图片及对应的标注文件按缺陷类别统计后，按比例随机分割到 train 和 val 文件夹
- 缺陷标注图少于20张的类别，全部加入 train
- 超过20张的类别，在同种缺陷内部按比例随机分配
"""

import os
import shutil
import random
import json


def get_image_extensions():
    return {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def get_annotation_extensions():
    return {'.xml', '.txt', '.json'}


def find_matching_annotations(image_name, annotation_dir):
    base_name = os.path.splitext(image_name)[0]
    found_annotations = []
    for ann_ext in get_annotation_extensions():
        ann_path = os.path.join(annotation_dir, base_name + ann_ext)
        if os.path.exists(ann_path):
            found_annotations.append(ann_path)
    return found_annotations


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
    image_primary_label = {}
    no_annotation_count = 0

    for image_name in images:
        json_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.json')
        if not os.path.exists(json_path):
            no_annotation_count += 1
            continue
        labels = load_json_labels(json_path)
        if not labels:
            no_annotation_count += 1
            continue
        for label in labels:
            label_image_count[label] = label_image_count.get(label, 0) + 1
        rarest = min(labels, key=lambda l: label_image_count.get(l, 0))
        image_primary_label[image_name] = rarest

    classified_images = {}
    for image_name, label in image_primary_label.items():
        classified_images.setdefault(label, []).append(image_name)

    return images, label_image_count, classified_images, no_annotation_count


def split_by_class(classified_images, train_ratio, min_threshold=20):
    train_images = []
    val_images = []
    class_stats = []

    for label, image_list in classified_images.items():
        random.shuffle(image_list)
        count = len(image_list)
        if count < min_threshold:
            if count == 1:
                train_images.extend(image_list)
                train_count = count
                val_count = 0
                print(f"  {label}: {count} 张 (< {min_threshold}) -> 全部归入 train")
            else:
                train_images.extend(image_list[:count - 1])
                val_images.append(image_list[count - 1])
                train_count = count - 1
                val_count = 1
                print(f"  {label}: {count} 张 (< {min_threshold}) -> train {count-1} 张, val 1 张")
        else:
            split_idx = int(count * train_ratio)
            train_images.extend(image_list[:split_idx])
            val_images.extend(image_list[split_idx:])
            train_count = split_idx
            val_count = count - split_idx
            print(f"  {label}: {count} 张 -> train: {train_count}, val: {val_count}")
        class_stats.append((label, count, train_count, val_count))

    return train_images, val_images, class_stats


def copy_files(image_list, image_dir, dest_dir):
    count = 0
    skipped_count = 0
    for image_name in image_list:
        image_src = os.path.join(image_dir, image_name)
        if not os.path.exists(image_src):
            continue
            
        # 检查是否有对应的标注文件
        ann_files = find_matching_annotations(image_name, image_dir)
        if not ann_files:
            skipped_count += 1
            continue
            
        # 复制图片
        shutil.copy2(image_src, dest_dir)
        
        # 复制标注文件
        for ann_path in ann_files:
            shutil.copy2(ann_path, dest_dir)
        
        count += 1
    
    if skipped_count > 0:
        print(f"  跳过 {skipped_count} 张无标注图片")
    return count


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
    images, label_image_count, classified_images, no_annotation_count = analyze_dataset(image_dir)

    if not images:
        print(f"错误: 在 '{image_dir}' 中未找到图片文件")
        return

    print(f"\n共找到 {len(images)} 张图片，{len(label_image_count)} 种缺陷类别:")
    if no_annotation_count > 0:
        print(f"其中 {no_annotation_count} 张图片无标注，将不会被复制到 train/val 目录")
    print("-" * 50)
    for label, count in sorted(label_image_count.items(), key=lambda x: x[1]):
        print(f"  {label}: {count} 张图片")
    print("-" * 50)

    classes_txt_path = os.path.join(image_dir, 'classes.txt')
    with open(classes_txt_path, 'w', encoding='utf-8') as f:
        for label in sorted(label_image_count.keys()):
            f.write(label + '\n')
    print(f"已生成类别文件: {classes_txt_path}")

    print(f"\n按缺陷类别分配:")
    print("-" * 50)
    train_images, val_images, class_stats = split_by_class(classified_images, train_ratio)
    print("-" * 50)

    train_dir = os.path.join(image_dir, 'train', 'images')
    val_dir = os.path.join(image_dir, 'val', 'images')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print(f"\n训练集: {len(train_images)} 张")
    print(f"验证集: {len(val_images)} 张")

    print("\n开始复制训练集...")
    train_count = copy_files(train_images, image_dir, train_dir)
    print(f"训练集复制完成: {train_count} 张图片")

    if val_images:
        print("开始复制验证集...")
        val_count = copy_files(val_images, image_dir, val_dir)
        print(f"验证集复制完成: {val_count} 张图片")

    print("\n完成!")
    print(f"训练集保存在: {train_dir}")
    print(f"验证集保存在: {val_dir}")

    print(f"\n{'='*60}")
    print(f"  各缺陷类别分配汇总")
    print(f"{'='*60}")
    print(f"  {'类别':<15} {'总计':>6} {'train':>8} {'val':>8}")
    print(f"  {'-'*42}")
    for label, total, t_count, v_count in sorted(class_stats, key=lambda x: x[1]):
        print(f"  {label:<15} {total:>6} {t_count:>8} {v_count:>8}")
    print(f"  {'-'*42}")
    print(f"  {'合计':<15} {sum(s[1] for s in class_stats):>6} {sum(s[2] for s in class_stats):>8} {sum(s[3] for s in class_stats):>8}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
