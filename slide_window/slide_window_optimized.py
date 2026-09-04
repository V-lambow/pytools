import os
import json
import numpy as np
from PIL import Image
import warnings
from concurrent.futures import ProcessPoolExecutor

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

def process_single_tile(args):
    tile_data, x, y, block_w, block_h, basename, r, c, out_dir, shapes_data, border_ext, img_version = args

    out_name = f"{basename}_r{r}_c{c}.png"
    out_path = os.path.join(out_dir, out_name)

    # tile_data是bytes，避免pickle大numpy数组
    img = Image.frombuffer('RGB' if block_w * 3 == len(tile_data) // block_h else 'L',
                           (block_w, block_h), tile_data,
                           'raw', 'RGB' if block_w * 3 == len(tile_data) // block_h else 'L', 0, 1)
    img.save(out_path)

    # 处理标注
    if shapes_data:
        new_shapes = []
        for shape in shapes_data:
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

            clipped, keep = clip_box(box, x, y, block_w, block_h)
            if keep:
                w = clipped[2] - clipped[0]
                h = clipped[3] - clipped[1]
                if w < 32 or h < 32:
                    continue
                new_shapes.append({**shape, "points": [
                    [clipped[0] - x, clipped[1] - y],
                    [clipped[2] - x, clipped[3] - y]
                ]})

        if new_shapes:
            out_ann_path = os.path.join(out_dir, f"{basename}_r{r}_c{c}.json")
            new_data = {
                "version": img_version, "flags": {}, "shapes": new_shapes,
                "imagePath": out_name, "imageData": None,
                "imageHeight": block_h, "imageWidth": block_w
            }
            with open(out_ann_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)

    return 1

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

    try:
        max_workers = int(input("请输入并行处理数 (推荐CPU核心数, 直接回车使用4): ").strip() or "4")
    except ValueError:
        max_workers = 4
    print(f"  并行数: {max_workers}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted([f.name for f in os.scandir(img_dir)
                    if f.is_file() and os.path.splitext(f.name)[1].lower() in exts])
    print(f"\n共找到 {len(files)} 张图片")

    all_tasks = []
    for fname in files:
        fpath = os.path.join(img_dir, fname)
        img = Image.open(fpath)
        img_np = np.asarray(img)
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

        # 每张图片的标注只读取一次
        shapes_data = []
        img_version = ""
        if os.path.exists(ann_path):
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                img_version = ann_data.get("version", "")
                shapes_data = [s for s in ann_data.get("shapes", [])
                               if s.get("shape_type") == "rectangle"]
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        for r in range(rows):
            for c in range(cols):
                y_pos = r * (block_h - overlap)
                x_pos = c * (block_w - overlap)
                if y_pos + block_h > pad_h or x_pos + block_w > pad_w:
                    continue
                # 传递tile的bytes而非numpy数组，避免pickle大数组
                tile = padded[y_pos:y_pos+block_h, x_pos:x_pos+block_w]
                tile_bytes = tile.tobytes()
                all_tasks.append((tile_bytes, x_pos, y_pos, block_w, block_h,
                                  basename, r, c, out_dir, shapes_data, border_ext, img_version))

        # 释放内存
        del padded, img_np, tile

    print(f"共 {len(all_tasks)} 个tile待处理，开始并行处理...")

    total_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for count in executor.map(process_single_tile, all_tasks):
            total_count += count

    print(f"\n全部完成! 共生成 {total_count} 个子块, 输出目录: {out_dir}")

if __name__ == "__main__":
    main()
