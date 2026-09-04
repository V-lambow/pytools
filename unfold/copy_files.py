import os
import shutil
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    HAS_GUI = True
except Exception:
    HAS_GUI = False


def gui_pick_folder() -> str:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title="请选择要遍历的文件夹")
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


if __name__ == "__main__":
    if HAS_GUI:
        folder = gui_pick_folder()
        if not folder:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("提示", "已取消")
            root.destroy()
            raise SystemExit
        try:
            flatten_copy(folder)
        except Exception as e:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("错误", str(e))
            root.destroy()
            raise SystemExit
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("完成", f"复制完成，文件已输出到 ./output")
        root.destroy()
    else:
        folder = input("请输入文件夹路径：").strip() or "."
        flatten_copy(folder)