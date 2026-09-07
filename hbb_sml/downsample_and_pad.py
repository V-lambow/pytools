#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片降采样脚本
对文件夹内的图片进行指定倍率降采样，同时对对应的json文件进行坐标变换
"""

import os
import json
from pathlib import Path
from PIL import Image


def process_image(image_path, output_path, downsample_ratio_x=2, downsample_ratio_y=2, output_format='png'):
    """处理单个图片：降采样"""
    img = Image.open(image_path)
    img.load()
    original_width, original_height = img.size

    new_width = original_width // downsample_ratio_x
    new_height = original_height // downsample_ratio_y

    downsampled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    downsampled_img.save(output_path, format=output_format.upper())

    img.close()
    downsampled_img.close()

    return {
        'original_size': (original_width, original_height),
        'new_size': (new_width, new_height),
        'downsample_ratio_x': downsample_ratio_x,
        'downsample_ratio_y': downsample_ratio_y
    }


def process_json(json_path, output_path, image_info):
    """处理单个json文件：坐标变换"""
    # 尝试不同编码读取JSON
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
    data = None
    
    for encoding in encodings:
        try:
            with open(json_path, 'r', encoding=encoding) as f:
                data = json.load(f)
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    
    if data is None:
        raise ValueError(f"无法读取JSON文件: {json_path}，请检查文件编码")

    downsample_ratio_x = image_info['downsample_ratio_x']
    downsample_ratio_y = image_info['downsample_ratio_y']

    data['imageWidth'] = image_info['new_size'][0]
    data['imageHeight'] = image_info['new_size'][1]

    if data.get('imageData'):
        data['imageData'] = None

    output_image_name = Path(output_path).name
    data['imagePath'] = output_image_name

    for shape in data.get('shapes', []):
        points = shape.get('points', [])
        new_points = []
        for point in points:
            x, y = point
            new_x = x / downsample_ratio_x
            new_y = y / downsample_ratio_y
            new_points.append([float(new_x), float(new_y)])
        shape['points'] = new_points

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 50)
    print("   图片降采样 + JSON坐标变换")
    print("=" * 50)

    input_dir_str = input("\n请输入输入文件夹路径: ").strip()
    downsample_ratio_x_str = input("请输入水平降采样倍率 (默认2): ").strip() or "2"
    downsample_ratio_y_str = input("请输入垂直降采样倍率 (默认2, 回车同水平): ").strip() or downsample_ratio_x_str
    output_format = input("请输入输出文件类型 (png/jpg/bmp/tiff, 默认png): ").strip() or "png"
    output_dir_str = input("请输入输出文件夹路径 (直接回车则自动生成): ").strip()

    downsample_ratio_x = int(downsample_ratio_x_str)
    downsample_ratio_y = int(downsample_ratio_y_str)
    if downsample_ratio_x < 1 or downsample_ratio_y < 1:
        print(f"错误: 降采样倍率必须 >= 1, 当前值: {downsample_ratio_x}x{downsample_ratio_y}")
        return

    valid_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']
    if output_format.lower() not in valid_formats:
        print(f"错误: 不支持的格式 '{output_format}'，可选: {valid_formats}")
        return

    input_dir = Path(input_dir_str)
    if not input_dir.exists():
        print(f"错误: 输入文件夹不存在: {input_dir}")
        return

    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = input_dir / f"downsampled_{downsample_ratio_x}x{downsample_ratio_y}y"

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.iterdir()
                  if f.is_file() and f.suffix.lower() in image_extensions]

    print(f"找到 {len(image_files)} 个图片文件")
    print(f"降采样倍率: {downsample_ratio_x}x{downsample_ratio_y}y")
    print(f"输出文件夹: {output_dir}")

    success_count = 0
    fail_count = 0
    for i, image_file in enumerate(image_files, 1):
        print(f"处理 [{i}/{len(image_files)}]: {image_file.name}", end="")

        try:
            output_filename = f"{image_file.stem}.{output_format}"
            output_path = output_dir / output_filename

            image_info = process_image(image_file, output_path, downsample_ratio_x, downsample_ratio_y, output_format)

            json_file = image_file.with_suffix('.json')
            if json_file.exists():
                json_output_path = output_dir / f"{image_file.stem}.json"
                process_json(json_file, json_output_path, image_info)

            success_count += 1
            print(" - OK")
        except Exception as e:
            fail_count += 1
            print(f" - FAILED: {e}")

    print(f"\n处理完成! 成功: {success_count}, 失败: {fail_count}")
    print(f"输出文件夹: {output_dir}")


if __name__ == "__main__":
    main()
