#!/bin/bash
# Linux打包脚本 - 在Ubuntu上运行

echo "========================================="
echo "滑动窗口处理工具 - Linux打包脚本"
echo "========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3，请先安装Python3"
    exit 1
fi

# 检查并安装依赖
echo "检查并安装依赖..."
pip3 install numpy pillow pyinstaller -q

# 打包
echo "开始打包..."
python3 -m PyInstaller --onefile --name slide_window_optimized slide_window_optimized.py

if [ $? -eq 0 ]; then
    echo ""
    echo "打包成功!"
    echo "可执行文件: dist/slide_window_optimized"
    echo "========================================="
    echo "运行方式:"
    echo "  ./dist/slide_window_optimized"
else
    echo "打包失败"
    exit 1
fi
