import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def calculate_area(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    return width * height

def analyze_labels(json_dir):
    label_counts = defaultdict(int)
    label_areas = defaultdict(list)
    file_count = 0

    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(json_dir, filename)
            file_count += 1
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data.get('shapes', []):
                    label = shape.get('label', 'unknown')
                    label_counts[label] += 1
                    if shape.get('shape_type') == 'rectangle':
                        area = calculate_area(shape.get('points', []))
                        label_areas[label].append(area)
            except Exception as e:
                print(f"  [!] Error processing {filename}: {e}")

    print(f"  Scanned {file_count} json files.")
    return label_counts, label_areas

def plot_bar_chart(label_counts, output_path):
    labels = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, counts, color='steelblue')

    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=10)

    plt.xlabel('Defect Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Defect Count Statistics', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_box_chart(label_areas, output_path):
    sorted_labels = sorted(label_areas.keys(), key=lambda x: len(label_areas[x]), reverse=True)
    data = [label_areas[label] for label in sorted_labels]

    plt.figure(figsize=(12, 6))
    box = plt.boxplot(data, patch_artist=True)
    plt.xticks(range(1, len(sorted_labels) + 1), sorted_labels, rotation=45, ha='right')

    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_labels)))
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Defect Category', fontsize=12)
    plt.ylabel('Area (pixels)', fontsize=12)
    plt.title('Defect Area Distribution Box Plot', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 50)
    print("       Label Statistics Tool")
    print("=" * 50)
    print()

    json_dir = input("Input: Enter label directory path (containing .json files): ").strip().strip('"')
    if not os.path.exists(json_dir):
        print(f"\n[ERROR] Directory not found: {json_dir}")
        input("\nPress Enter to exit...")
        return

    output_dir = input("Output: Enter output directory path: ").strip().strip('"')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  Input  : {json_dir}")
    print(f"  Output : {output_dir}")
    print()
    print("Analyzing label files...")

    label_counts, label_areas = analyze_labels(json_dir)

    if not label_counts:
        print("\n[ERROR] No label data found!")
        input("\nPress Enter to exit...")
        return

    print(f"\nAnalysis complete!")
    print(f"  Total categories: {len(label_counts)}")
    print(f"  Total defects  : {sum(label_counts.values())}")
    print()
    print("  Category        Count     Avg Area(px)")
    print("  " + "-" * 40)
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        areas = label_areas.get(label, [])
        avg_area = np.mean(areas) if areas else 0
        print(f"  {label:<15} {count:<9} {avg_area:.0f}")

    bar_path = os.path.join(output_dir, "defect_count_statistics.png")
    box_path = os.path.join(output_dir, "defect_area_statistics.png")

    print("\nGenerating charts...")
    plot_bar_chart(label_counts, bar_path)
    print(f"  [OK] Count chart: {bar_path}")
    plot_box_chart(label_areas, box_path)
    print(f"  [OK] Area  chart: {box_path}")

    print("\nDone!")
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")