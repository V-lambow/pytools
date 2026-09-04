import os
import json
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

def clip_box(box, tile_x, tile_y, tile_w, tile_h):
    x1, y1, x2, y2 = box
    cx1 = max(x1, tile_x)
    cy1 = max(y1, tile_y)
    cx2 = min(x2, tile_x + tile_w)
    cy2 = min(y2, tile_y + tile_h)
    if cx1 >= cx2 or cy1 >= cy2:
        return None, False
    return [cx1, cy1, cx2, cy2], True

def process_annotation(ann_path, tile_x, tile_y, tile_w, tile_h, border_ext, new_image_name, overlap=32):
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    new_shapes = []
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        pts = shape["points"]
        # 兼容2点和4点格式
        all_x = [p[0] for p in pts]
        all_y = [p[1] for p in pts]
        box = [min(all_x), min(all_y), max(all_x), max(all_y)]

        if border_ext > 0:
            box[0] += border_ext; box[1] += border_ext
            box[2] += border_ext; box[3] += border_ext
        elif border_ext < 0:
            shrink = -border_ext
            box[0] -= shrink; box[1] -= shrink
            box[2] -= shrink; box[3] -= shrink

        clipped, keep = clip_box(box, tile_x, tile_y, tile_w, tile_h)
        if not keep:
            continue

        # 只有被当前子块边界切分，并且切分后宽或高任一小于
        # 交叠像素的一半时，才在当前小图中删除该标注。
        # 未被切分的框，即使尺寸很小，也必须保留。
        on_edge = (clipped[0] != box[0] or clipped[1] != box[1] or
                   clipped[2] != box[2] or clipped[3] != box[3])
        if on_edge:
            clipped_w = clipped[2] - clipped[0]
            clipped_h = clipped[3] - clipped[1]
            if clipped_w < overlap / 2 or clipped_h < overlap / 2:
                continue

        new_shapes.append({**shape, "points": [
            [clipped[0] - tile_x, clipped[1] - tile_y],
            [clipped[2] - tile_x, clipped[3] - tile_y]
        ]})

    if not new_shapes:
        return None

    return {
        "version": data.get("version", ""),
        "flags": {},
        "shapes": new_shapes,
        "imagePath": new_image_name,
        "imageData": None,
        "imageHeight": tile_h,
        "imageWidth": tile_w,
    }

def main():
    img_dir = input("请输入数据集目录路径: ").strip()
    if not os.path.isdir(img_dir):
        print(f"错误: 目录不存在 - {img_dir}"); return

    grid_input = input("请输入要分割成几行几列 (格式: 行数,列数): ").strip()
    try:
        rows, cols = map(int, grid_input.split(","))
        print(f"  行数: {rows}, 列数: {cols}")
    except ValueError:
        print("错误: 行列数格式不正确"); return

    try:
        overlap = int(input("请输入不同框之间的交叠像素: ").strip())
        print(f"  交叠像素: {overlap}")
    except ValueError:
        print("错误: 交叠像素必须是整数"); return

    border_input = input("请输入整图的边框扩展像素 (+为向外扩充, -为向内缩减, 0为不变): ").strip()
    if not border_input:
        print("错误: 请输入边框扩展像素"); return
    if border_input == "0":
        border_ext = 0
    elif border_input[0] in ("+", "-"):
        try: border_ext = int(border_input)
        except ValueError: print("错误: 边框扩展像素必须是数字"); return
    else:
        print("错误: 边框扩展格式应以 +、- 开头或输入0"); return
    print(f"  边框扩展: {border_ext} 像素")

    out_dir = input("请输入输出目录路径: ").strip()
    os.makedirs(out_dir, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([f.name for f in os.scandir(img_dir)
                    if f.is_file() and os.path.splitext(f.name)[1].lower() in exts])
    print(f"\n共找到 {len(files)} 张图片")

    for fname in files:
        fpath = os.path.join(img_dir, fname)
        img = Image.open(fpath)
        img_np = np.asarray(img)  # 零拷贝: 共享PIL内存
        orig_h, orig_w = img_np.shape[:2]

        if border_ext > 0:
            padded = np.zeros((orig_h + border_ext*2, orig_w + border_ext*2, img_np.shape[2]),
                              dtype=img_np.dtype) if img_np.ndim == 3 else \
                     np.zeros((orig_h + border_ext*2, orig_w + border_ext*2), dtype=img_np.dtype)
            padded[border_ext:border_ext+orig_h, border_ext:border_ext+orig_w] = img_np
        elif border_ext < 0:
            shrink = -border_ext
            if shrink * 2 >= orig_h or shrink * 2 >= orig_w:
                print(f"  警告: {fname} 缩减像素过大，跳过"); continue
            padded = img_np[shrink:orig_h-shrink, shrink:orig_w-shrink].copy()
        else:
            padded = img_np

        pad_h, pad_w = padded.shape[:2]
        block_w = int((pad_w + overlap * (cols - 1)) / cols)
        block_h = int((pad_h + overlap * (rows - 1)) / rows)
        basename = os.path.splitext(fname)[0]
        ann_path = os.path.join(img_dir, basename + ".json")

        split_count = 0
        annotated_count = 0
        skipped_count = 0
        annotation_file_exists = os.path.exists(ann_path)
        for r in range(rows):
            for c in range(cols):
                y = r * (block_h - overlap)
                x = c * (block_w - overlap)
                if y + block_h > pad_h or x + block_w > pad_w:
                    continue

                split_count += 1
                tile = padded[y:y+block_h, x:x+block_w]  # 零拷贝: view
                out_name = f"{basename}_r{r}_c{c}.png"

                if os.path.exists(ann_path):
                    ann_data = process_annotation(ann_path, x, y, block_w, block_h, border_ext, out_name, overlap)
                    if ann_data is None:
                        skipped_count += 1
                        continue

                    annotated_count += 1
                    out_ann_path = os.path.join(out_dir, f"{basename}_r{r}_c{c}.json")
                    with open(out_ann_path, 'w', encoding='utf-8') as f:
                        json.dump(ann_data, f, ensure_ascii=False, indent=2)

                    Image.frombuffer('RGB' if tile.ndim == 3 else 'L',
                                     (block_w, block_h), tile.tobytes(),
                                     'raw', 'RGB' if tile.ndim == 3 else 'L', 0, 1).save(
                        os.path.join(out_dir, out_name))
                else:
                    skipped_count += 1

        print(f"  {fname}: 切割子块 {split_count}/{rows * cols} 个, "
              f"有框子块 {annotated_count} 个, 无框跳过 {skipped_count} 个, "
              f"标注文件{'存在' if annotation_file_exists else '不存在'}")

    print(f"\n全部完成! 输出目录: {out_dir}")

if __name__ == "__main__":
    main()
