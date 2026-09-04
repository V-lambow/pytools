#!/usr/bin/env python3
"""跨平台打包脚本 - 在目标OS上运行"""
import subprocess
import sys
import platform

def main():
    os_name = platform.system()
    print("=" * 50)
    print(f"当前系统: {os_name}")
    print("=" * 50)

    # 检查依赖
    print("安装依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "pillow", "pyinstaller", "-q"], check=True)

    # 打包
    name = "slide_window_optimized"
    if os_name == "Windows":
        name += ".exe"

    print(f"打包中 -> dist/{name}")
    subprocess.run([
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "slide_window_optimized",
        "slide_window_optimized.py"
    ], check=True)

    print(f"\n打包完成: dist/{name}")

if __name__ == "__main__":
    main()
