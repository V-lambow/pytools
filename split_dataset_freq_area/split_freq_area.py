"""
数据集按[出现频次 + HBB框大小]二维KMeans聚类分割工具

功能：
1. 遍历所有标注，统计每个标签的出现图片数(频次) 和 平均HBB面积
2. 每张图取全局出现次数最少的标签为主标签
3. 对每个标签构建二维特征 [log1p(频次), log1p(平均面积)]，标准化后做KMeans
4. 聚类数K不再固定为2：
   - 脚本先计算数据分布，用轮廓系数(Silhouette)推荐最优K
   - 在命令行让使用者确认推荐的K；否定则输入自定义K
5. 按输入比例随机拆分train/val（主标签为准，<20张全进train）
6. train和val中，根据图片所有标签的聚类归属分发到对应组：
   - 全部标签在同一组 -> 只进该组
   - 跨多组 -> 同时进多个组，各组只保留对应聚类的标签
7. 输出到输入文件夹内，生成各组的classes.txt和yaml配置文件

输出结构：
  g0/train/images/
  g0/val/images/
  g1/train/images/
  g1/val/images/
  ...
"""

import os
import sys
import shutil
import random
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


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
    skipped_images = []

    for image_name in images:
        json_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.json')
        if not os.path.exists(json_path):
            skipped_images.append(image_name)
            continue
        labels, boxes = load_json_labels_and_boxes(json_path)
        if not labels:
            skipped_images.append(image_name)
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

    if label_mean_area:
        fallback_area = float(np.median(list(label_mean_area.values())))
    else:
        fallback_area = 1.0

    mean_areas = {}
    for label in label_image_count:
        if label in label_mean_area:
            mean_areas[label] = label_mean_area[label]
        else:
            mean_areas[label] = fallback_area

    image_total_area = {}
    for image_name, labels in image_labels.items():
        total = 0
        for label in labels:
            json_path = os.path.join(image_dir, os.path.splitext(image_name)[0] + '.json')
            if os.path.exists(json_path):
                _, boxes = load_json_labels_and_boxes(json_path)
                if label in boxes:
                    total += sum(boxes[label])
        image_total_area[image_name] = total

    classified_images = {}
    for image_name, label in image_primary_label.items():
        classified_images.setdefault(label, []).append(image_name)

    return (images, label_image_count, mean_areas, image_labels,
            image_primary_label, classified_images, skipped_images, image_total_area)


def build_features(label_image_count, mean_areas):
    labels = sorted(label_image_count.keys())
    freq = np.array([label_image_count[l] for l in labels]).reshape(-1, 1)
    area = np.array([mean_areas[l] for l in labels]).reshape(-1, 1)
    X = np.hstack([np.log1p(freq), np.log1p(area)])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return labels, X_scaled, X


def recommend_k(X, max_k=10):
    n = X.shape[0]
    if n < 2:
        return 1, {}
    k_max = min(max_k, n - 1)
    k_range = list(range(2, k_max + 1))
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        labels = km.labels_
        if len(set(labels)) > 1:
            sil = float(silhouette_score(X, labels))
        else:
            sil = -1.0
        scores[k] = (sil, float(km.inertia_))
    best_k = max(scores, key=lambda k: scores[k][0])
    return best_k, scores


def ask_k(best_k, scores, n_labels):
    print("\n各K值聚类效果评估 (Silhouette 越高越好, Inertia 越低越好):")
    print("-" * 60)
    print(f"  {'K':>3} {'Silhouette':>12} {'Inertia':>14}")
    print("-" * 60)
    for k, (sil, inert) in sorted(scores.items()):
        mark = "  <- 推荐" if k == best_k else ""
        print(f"  {k:>3} {sil:>12.4f} {inert:>14.1f}{mark}")
    print("-" * 60)

    while True:
        sys.stdout.flush()
        try:
            ans = input(f"\n推荐聚类数 K = {best_k}\n"
                        f"  直接回车 或输入 y  -> 采用推荐值 {best_k}\n"
                        f"  输入其他数字       -> 使用你自定义的聚类数\n"
                        f"  请做出选择: ").strip()
        except EOFError:
            print(f"\n未检测到有效输入，使用推荐值 K = {best_k}")
            return best_k
        print()
        if ans in ('', 'y', 'Y', 'yes', 'YES'):
            print(f"  已确认使用推荐值 K = {best_k}")
            return best_k
        try:
            k = int(ans)
        except ValueError:
            print(f"  输入无效: '{ans}'，请重新输入 (数字K或y)")
            continue
        if k < 2:
            print("  K 必须 >= 2，请重新输入")
            continue
        if k > n_labels:
            print(f"  K 不能超过标签总数 {n_labels}，请重新输入")
            continue
        print(f"  已确认自定义 K = {k}")
        return k


def cluster_labels(X, k, labels):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    cluster_of_label = {label: int(cid) for label, cid in zip(labels, kmeans.labels_)}
    centers = kmeans.cluster_centers_

    order = sorted(range(k), key=lambda c: centers[c][1], reverse=True)
    new_id = {old: new for new, old in enumerate(order)}

    group_labels = [set() for _ in range(k)]
    for label, old_cid in cluster_of_label.items():
        group_labels[new_id[old_cid]].add(label)

    return group_labels, kmeans, cluster_of_label


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


def split_by_class_area_priority(classified_images, image_areas, train_ratio):
    """
    使用LUT方式：按面积排序后，前后各一半的train比例进train，中间进val
    例如 train:val = 8:2 -> 前40% + 后40% 进train，中间20% 进val
    """
    all_images = []
    for label, images in classified_images.items():
        for img in images:
            area = image_areas.get(img, 0)
            all_images.append((img, area))

    all_images.sort(key=lambda x: x[1])

    n = len(all_images)
    half_train = train_ratio / 2

    train_images = []
    val_images = []

    split_start = int(n * half_train)
    split_end = int(n * (1 - half_train))

    for item in all_images[:split_start]:
        train_images.append(item[0])

    for item in all_images[split_end:]:
        train_images.append(item[0])

    for item in all_images[split_start:split_end]:
        val_images.append(item[0])

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
    (images, label_image_count, mean_areas, image_labels,
     image_primary_label, classified_images, skipped_images, image_total_area) = analyze_dataset(image_dir)

    if not images:
        print(f"错误: 在 '{image_dir}' 中未找到图片文件")
        return

    print(f"\n共找到 {len(images)} 张图片，{len(label_image_count)} 种标签:")
    if skipped_images:
        print(f"  其中 {len(skipped_images)} 张图片因无标注而跳过")
    print("-" * 70)
    print(f"  {'标签':<15} {'出现图片数':>10} {'平均HBB面积':>14}")
    print("-" * 70)
    for label, count in sorted(label_image_count.items(), key=lambda x: x[1]):
        print(f"  {label:<15} {count:>10} {mean_areas.get(label, 0):>14.1f}")
    print("-" * 70)

    labels, X_scaled, X = build_features(label_image_count, mean_areas)

    if len(labels) < 2:
        print("\n错误: 标签数量不足2个，无法聚类")
        return

    print("\n正在评估数据分布以推荐最优K...")
    best_k, scores = recommend_k(X_scaled)
    k = ask_k(best_k, scores, len(labels))

    print(f"\n正在进行[频次+面积]二维KMeans聚类 (K={k})...")
    group_labels, kmeans, cluster_of_label = cluster_labels(X_scaled, k, labels)

    for gi, labels_in_group in enumerate(group_labels):
        print(f"\n  g{gi} 组标签 ({len(labels_in_group)}个):")
        for l in sorted(labels_in_group, key=lambda x: label_image_count[x], reverse=True):
            print(f"    {l}: 频次 {label_image_count[l]} 张, 平均面积 {mean_areas.get(l, 0):.1f}")

    print(f"\n按面积优先分配 train/val (极端面积优先进训练集):")
    print("-" * 70)
    train_images, val_images = split_by_class_area_priority(classified_images, image_total_area, train_ratio)
    print(f"  train: {len(train_images)} 张, val: {len(val_images)} 张")
    print("-" * 70)

    group_dirs = []
    for gi in range(k):
        train_dir = os.path.join(image_dir, f'g{gi}', 'train', 'images')
        val_dir = os.path.join(image_dir, f'g{gi}', 'val', 'images')
        for d in [train_dir, val_dir]:
            os.makedirs(d, exist_ok=True)
        group_dirs.append((train_dir, val_dir))

    group_train_count = [0] * k
    group_val_count = [0] * k

    print(f"\n正在分配训练集到各聚类组...")
    for image_name in train_images:
        labels_in_image = image_labels.get(image_name, set())
        for gi in range(k):
            if labels_in_image & group_labels[gi]:
                copy_with_trimmed_annotation(image_dir, image_name, group_dirs[gi][0], group_labels[gi])
                group_train_count[gi] += 1
    print("  " + ", ".join(f"g{gi}: {group_train_count[gi]} 张" for gi in range(k)))

    print(f"\n正在分配验证集到各聚类组...")
    for image_name in val_images:
        labels_in_image = image_labels.get(image_name, set())
        for gi in range(k):
            if labels_in_image & group_labels[gi]:
                copy_with_trimmed_annotation(image_dir, image_name, group_dirs[gi][1], group_labels[gi])
                group_val_count[gi] += 1
    print("  " + ", ".join(f"g{gi}: {group_val_count[gi]} 张" for gi in range(k)))

    folder_name = os.path.basename(os.path.normpath(image_dir))

    for gi in range(k):
        write_classes_txt(os.path.join(image_dir, f'g{gi}', 'classes.txt'), group_labels[gi])
        write_yaml(os.path.join(image_dir, f'{folder_name}_g{gi}.yaml'),
                   image_dir, f'g{gi}/train/images', f'g{gi}/val/images', group_labels[gi])

    print(f"\n{'='*70}")
    print(f"  分割结果汇总 (K = {k})")
    print(f"{'='*70}")
    print(f"  总图片数: {len(images)}")
    if skipped_images:
        print(f"  跳过无标注图片: {len(skipped_images)} 张")
    print(f"  参与分类图片数: {len(images) - len(skipped_images)}")
    print(f"  train原始: {len(train_images)}, val原始: {len(val_images)}")
    for gi in range(k):
        print(f"  g{gi}组 train: {group_train_count[gi]}, val: {group_val_count[gi]} ({len(group_labels[gi])}个标签)")
    print(f"{'='*70}")
    print(f"\n输出目录: {image_dir}")
    for gi in range(k):
        print(f"  g{gi}/train/images/     g{gi}/classes.txt")
        print(f"  g{gi}/val/images/")
        print(f"  {folder_name}_g{gi}.yaml ({len(group_labels[gi])} 个标签)")
    print(f"\n完成!")


if __name__ == "__main__":
    main()
