#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片降采样和填充脚本
对文件夹内的图片进行一倍降采样，四周使用gray0填充到原来的大小
同时对对应的json文件进行坐标变换
"""

import os
import json
from pathlib import Path
from PIL import Image


def process_image(image_path, output_path, output_format='png'):
    """处理单个图片：降采样并填充"""
    img = Image.open(image_path)
    img.load()  # 强制加载数据，提前暴露问题
    original_width, original_height = img.size
    
    # 计算降采样后的尺寸（缩小一半）
    new_width = original_width // 2
    new_height = original_height // 2
    
    # 降采样
    downsampled_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # 创建黑色背景（gray0）
    if img.mode == 'L':
        padded_img = Image.new('L', (original_width, original_height), 0)
    else:
        padded_img = Image.new('RGB', (original_width, original_height), (0, 0, 0))
    
    # 计算居中放置的偏移量
    x_offset = (original_width - new_width) // 2
    y_offset = (original_height - new_height) // 2
    
    # 将降采样后的图片粘贴到黑色背景上
    padded_img.paste(downsampled_img, (x_offset, y_offset))
    
    # 保存图片
    padded_img.save(output_path, format=output_format.upper())
    
    img.close()
    downsampled_img.close()
    padded_img.close()
    
    return {
        'original_size': (original_width, original_height),
        'new_size': (new_width, new_height),
        'offset': (x_offset, y_offset)
    }


def process_json(json_path, output_path, image_info):
    """处理单个json文件：坐标变换"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取偏移量和缩放信息
    x_offset, y_offset = image_info['offset']
    original_width, original_height = image_info['original_size']
    new_width, new_height = image_info['new_size']
    
    # 计算缩放比例
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    
    # 处理每个形状
    for shape in data.get('shapes', []):
        if shape.get('shape_type') == 'rectangle':
            # 矩形有4个点
            points = shape['points']
            if len(points) == 4:
                new_points = []
                for point in points:
                    x, y = point
                    # 1. 缩放坐标
                    new_x = x * scale_x
                    new_y = y * scale_y
                    # 2. 添加偏移量
                    new_x += x_offset
                    new_y += y_offset
                    new_points.append([new_x, new_y])
                shape['points'] = new_points
        elif shape.get('shape_type') == 'polygon':
            # 多边形可能有多个点
            points = shape['points']
            new_points = []
            for point in points:
                x, y = point
                # 1. 缩放坐标
                new_x = x * scale_x
                new_y = y * scale_y
                # 2. 添加偏移量
                new_x += x_offset
                new_y += y_offset
                new_points.append([new_x, new_y])
            shape['points'] = new_points
    
    # 保存json文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 50)
    print("   图片降采样 + 填充 + JSON坐标变换")
    print("=" * 50)
    
    # 交互式输入
    input_dir_str = input("\n请输入输入文件夹路径: ").strip()
    output_format = input("请输入输出文件类型 (png/jpg/bmp/tiff, 默认png): ").strip() or "png"
    output_dir_str = input("请输入输出文件夹路径 (直接回车则自动生成): ").strip()
    
    # 校验输出格式
    valid_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']
    if output_format.lower() not in valid_formats:
        print(f"错误: 不支持的格式 '{output_format}'，可选: {valid_formats}")
        return
    
    input_dir = Path(input_dir_str)
    if not input_dir.exists():
        print(f"错误: 输入文件夹不存在: {input_dir}")
        return
    
    # 确定输出文件夹
    if output_dir_str:
        output_dir = Path(output_dir_str)
    else:
        output_dir = input_dir.parent / f"{input_dir.name}_downsampled"
    
    # 创建输出文件夹
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in input_dir.iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"找到 {len(image_files)} 个图片文件")
    print(f"输出文件夹: {output_dir}")
    print(f"输出格式: {output_format}")
    
    # 处理每个图片
    success_count = 0
    fail_count = 0
    for i, image_file in enumerate(image_files, 1):
        print(f"处理 [{i}/{len(image_files)}]: {image_file.name}", end="")
        
        try:
            # 确定输出文件名
            output_filename = f"{image_file.stem}.{output_format}"
            output_path = output_dir / output_filename
            
            # 处理图片
            image_info = process_image(image_file, output_path, output_format)
            
            # 处理对应的json文件
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
