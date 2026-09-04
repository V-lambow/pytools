import os
import json
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def clip_box(box, tile_x, tile_y, tile_w, tile_h):
    """裁剪标注框到tile范围内，返回裁剪后的框和是否保留"""
    x1, y1, x2, y2 = box
    cx1 = max(x1, tile_x)
    cy1 = max(y1, tile_y)
    cx2 = min(x2, tile_x + tile_w)
    cy2 = min(y2, tile_y + tile_h)
    if cx1 >= cx2 or cy1 >= cy2:
        return None, False
    return [cx1, cy1, cx2, cy2], True

def process_annotation(ann_path, out_path, tile_x, tile_y, tile_w, tile_h, border_ext, new_image_name):
    """处理单个标注文件，裁剪到tile范围内。返回是否有标签"""
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    new_shapes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") == "rectangle":
            pts = shape["points"]
            # 兼容2点和4点格式
            all_x = [p[0] for p in pts]
            all_y = [p[1] for p in pts]
            box = [min(all_x), min(all_y), max(all_x), max(all_y)]

            # 考虑边框扩展的影响
            if border_ext > 0:
                box[0] += border_ext
                box[1] += border_ext
                box[2] += border_ext
                box[3] += border_ext
            elif border_ext < 0:
                shrink = -border_ext
                box[0] -= shrink
                box[1] -= shrink
                box[2] -= shrink
                box[3] -= shrink

            clipped, keep = clip_box(box, tile_x, tile_y, tile_w, tile_h)
            if keep:
                new_shape = shape.copy()
                new_shape["points"] = [
                    [clipped[0], clipped[1]],
                    [clipped[2], clipped[3]]
                ]
                new_shapes.append(new_shape)

    if not new_shapes:
        return False

    new_data = {
        "version": data.get("version", ""),
        "flags": {},
        "shapes": new_shapes,
        "imagePath": new_image_name,
        "imageData": None,
        "imageHeight": tile_h,
        "imageWidth": tile_w
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    return True

def main():
    # 1. 数据集目录
    img_dir = input("请输入数据集目录路径: ").strip()
    if not os.path.isdir(img_dir):
        print(f"错误: 目录不存在 - {img_dir}")
        return

    # 2. 行列数
    grid_input = input("请输入要分割成几行几列 (格式: 行数,列数): ").strip()
    try:
        rows, cols = map(int, grid_input.split(","))
        print(f"  行数: {rows}, 列数: {cols}")
    except ValueError:
        print("错误: 行列数格式不正确，请使用逗号分隔的两个数字")
        return

    # 3. 交叠像素
    try:
        overlap = int(input("请输入不同框之间的交叠像素: ").strip())
        print(f"  交叠像素: {overlap}")
    except ValueError:
        print("错误: 交叠像素必须是整数")
        return

    # 4. 边框扩展像素
    border_input = input("请输入整图的边框扩展像素 (+为向外扩充, -为向内缩减, 0为不变): ").strip()
    if not border_input:
        print("错误: 请输入边框扩展像素")
        return
    if border_input == "0":
        border_ext = 0
    elif border_input[0] in ("+", "-"):
        try:
            border_ext = int(border_input)
        except ValueError:
            print("错误: 边框扩展像素必须是数字")
            return
    else:
        print("错误: 边框扩展格式应以 +、- 开头或输入0")
        return
    print(f"  边框扩展: {border_ext} 像素")

    # 5. 输出目录
    out_dir = input("请输入输出目录路径: ").strip()
    os.makedirs(out_dir, exist_ok=True)

    # 获取图片文件列表
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in exts]
    files.sort()
    print(f"\n共找到 {len(files)} 张图片")

    for fname in files:
        fpath = os.path.join(img_dir, fname)
        img = Image.open(fpath)
        img_np = np.array(img)
        orig_h, orig_w = img_np.shape[:2]

        # 处理边框扩展/缩减
        if border_ext > 0:
            pad_top = pad_bottom = pad_left = pad_right = border_ext
            if img_np.ndim == 3:
                padded = np.zeros((orig_h + pad_top + pad_bottom,
                                   orig_w + pad_left + pad_right,
                                   img_np.shape[2]), dtype=img_np.dtype)
            else:
                padded = np.zeros((orig_h + pad_top + pad_bottom,
                                   orig_w + pad_left + pad_right), dtype=img_np.dtype)
            padded[pad_top:pad_top+orig_h, pad_left:pad_left+orig_w] = img_np
        elif border_ext < 0:
            shrink = -border_ext
            if shrink * 2 >= orig_h or shrink * 2 >= orig_w:
                print(f"  警告: {fname} 缩减像素过大，跳过")
                continue
            padded = img_np[shrink:orig_h-shrink, shrink:orig_w-shrink]
        else:
            padded = img_np

        pad_h, pad_w = padded.shape[:2]

        # 计算每个子块的尺寸
        block_w = int((pad_w + overlap * (cols - 1)) / cols)
        block_h = int((pad_h + overlap * (rows - 1)) / rows)

        # 滑窗切割
        basename = os.path.splitext(fname)[0]
        ann_basename = basename
        count = 0
        for r in range(rows):
            for c in range(cols):
                y = r * (block_h - overlap)
                x = c * (block_w - overlap)

                # 边界检查
                if y + block_h > pad_h or x + block_w > pad_w:
                    continue

                tile = padded[y:y+block_h, x:x+block_w]
                out_name = f"{basename}_r{r}_c{c}.png"

                # 处理对应的标注文件
                ann_name = ann_basename + ".json"
                ann_path = os.path.join(img_dir, ann_name)
                has_annotation = False
                if os.path.exists(ann_path):
                    out_ann_name = f"{ann_basename}_r{r}_c{c}.json"
                    out_ann_path = os.path.join(out_dir, out_ann_name)
                    has_annotation = process_annotation(ann_path, out_ann_path, x, y, block_w, block_h, border_ext, out_name)

                if has_annotation:
                    Image.fromarray(tile).save(os.path.join(out_dir, out_name))

                count += 1

        print(f"  {fname}: 完成, 共生成 {count} 个子块")

    print(f"\n全部完成! 输出目录: {out_dir}")

if __name__ == "__main__":
    main()
