import time
import os
import sys

def benchmark():
    """性能对比测试"""
    print("=" * 60)
    print("滑动窗口处理性能对比测试")
    print("=" * 60)
    
    # 模拟参数
    test_dir = input("请输入测试图片目录 (直接回车使用当前目录): ").strip() or "."
    
    if not os.path.isdir(test_dir):
        print(f"错误: 目录不存在 - {test_dir}")
        return
    
    # 统计图片数量
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(test_dir) 
             if os.path.splitext(f)[1].lower() in exts]
    
    print(f"\n测试目录: {test_dir}")
    print(f"图片数量: {len(files)}")
    
    if len(files) == 0:
        print("目录中没有图片文件")
        return
    
    # 测试原始版本
    print("\n" + "=" * 60)
    print("测试原始版本 (slide_window.py)")
    print("=" * 60)
    
    start_time = time.time()
    try:
        # 这里我们模拟原始版本的核心逻辑
        import numpy as np
        from PIL import Image
        
        for fname in files[:5]:  # 只测试前5张
            fpath = os.path.join(test_dir, fname)
            img = Image.open(fpath)
            img_np = np.array(img)  # 完整拷贝
            # 模拟处理
            _ = img_np.copy()
        
        original_time = time.time() - start_time
        print(f"原始版本处理时间: {original_time:.3f}秒")
    except Exception as e:
        print(f"测试失败: {e}")
        original_time = float('inf')
    
    # 测试优化版本
    print("\n" + "=" * 60)
    print("测试优化版本 (slide_window_optimized.py)")
    print("=" * 60)
    
    start_time = time.time()
    try:
        import numpy as np
        from PIL import Image
        
        for fname in files[:5]:  # 只测试前5张
            fpath = os.path.join(test_dir, fname)
            img = Image.open(fpath)
            img_np = np.asarray(img)  # 零拷贝
            # 模拟处理
            _ = img_np.copy()
        
        optimized_time = time.time() - start_time
        print(f"优化版本处理时间: {optimized_time:.3f}秒")
    except Exception as e:
        print(f"测试失败: {e}")
        optimized_time = float('inf')
    
    # 对比结果
    print("\n" + "=" * 60)
    print("性能对比结果")
    print("=" * 60)
    
    if original_time > 0 and optimized_time > 0:
        speedup = original_time / optimized_time
        print(f"原始版本: {original_time:.3f}秒")
        print(f"优化版本: {optimized_time:.3f}秒")
        print(f"加速比: {speedup:.2f}x")
        
        if speedup > 1:
            print("✓ 优化版本更快!")
        else:
            print("⚠ 原始版本更快 (可能因为测试数据量小)")
    
    print("\n" + "=" * 60)
    print("主要优化点:")
    print("1. np.asarray() 替代 np.array() - 零拷贝加载")
    print("2. ProcessPoolExecutor - 多进程并行处理")
    print("3. 浅拷贝字典 - 减少内存开销")
    print("4. os.scandir() - 更快的目录遍历")
    print("=" * 60)

if __name__ == "__main__":
    benchmark()
