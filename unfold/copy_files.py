import argparse
import os
import shutil
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    HAS_GUI = True
except Exception:
    HAS_GUI = False


def gui_pick_folder(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path


def unique_path(dest_dir: Path, name: str) -> Path:
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = dest_dir / name
    counter = 1
    while candidate.exists():
        candidate = dest_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def flatten_copy(src_root: str, output_dir: str = "./output"):
    src_root = Path(src_root).resolve()
    dest_dir = Path(output_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for root, dirs, files in os.walk(src_root):
        if Path(root).resolve() == dest_dir:
            continue
        for file in files:
            src = Path(root) / file
            dst = unique_path(dest_dir, file)
            shutil.copy2(src, dst)
            copied += 1
    print(f"完成：共复制 {copied} 个文件到 {dest_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="递归文件夹展平复制工具")
    parser.add_argument("-i", "--input", help="源文件夹路径")
    parser.add_argument("-o", "--output", default="./output", help="输出文件夹路径（默认 ./output）")
    return parser.parse_args()


def main():
    args = parse_args()

    src = args.input
    dst = args.output

    if not src:
        if HAS_GUI:
            src = gui_pick_folder("请选择要遍历的文件夹")
            if not src:
                root = tk.Tk()
                root.withdraw()
                messagebox.showinfo("提示", "已取消")
                root.destroy()
                raise SystemExit
        else:
            src = input("请输入源文件夹路径：").strip() or "."

    if dst == "./output" and HAS_GUI and not args.input:
        output = gui_pick_folder("请选择输出文件夹（留空使用 ./output）")
        if output:
            dst = output

    try:
        flatten_copy(src, dst)
    except Exception as e:
        if HAS_GUI:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("错误", str(e))
            root.destroy()
        else:
            print(f"错误：{e}")
        raise SystemExit

    if HAS_GUI:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("完成", f"复制完成，文件已输出到 {dst}")
        root.destroy()


if __name__ == "__main__":
    main()
