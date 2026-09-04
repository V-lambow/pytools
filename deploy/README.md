# detect_server.py 服务文档

## 项目概述

基于 **Tornado Web 框架 + Ultralytics YOLO** 的工业图像缺陷检测 REST API 服务。  
监听端口 `8002`，提供 `POST /industry/image_defect` 接口，对工业产品图像进行目标检测并返回缺陷结果。

---

## 架构设计

```
HTTP Client
    │
    ▼
Tornado Server (port 8002)
    │
    ▼
POST /industry/image_defect
    ├── 1. 解析请求参数 (job_id, sample_id, pose_id, file_names, relative_dir)
    ├── 2. 根据 pose_id 加载区域配置 (assemble_detect_item.json)
    ├── 3. 按 region 裁剪 ROI 区域
    ├── 4. YOLO 模型推理 (ThreadPoolExecutor, 线程安全)
    ├── 5. 判断置信度阈值，返回检测结果 JSON
    └── 6. 异步绘制缺陷框并保存到 res/ 目录
```

---

## 功能模块

### 1. `ImageDefectHandler` — 主请求处理器

| 方法 | 行号 | 功能 |
|------|------|------|
| `initialize()` | L42-67 | 初始化配置目录、加载 YOLO 模型、创建线程池和模型锁 |
| `load_config()` | L69-76 | 读取 `assemble_detect_item.json` 配置文件 |
| `post()` | L80-133 | HTTP POST 入口，解析参数、调用检测、返回结果 |
| `process_image()` | L135-217 | 核心检测逻辑：按区域裁剪 → YOLO 推理 → 返回缺陷列表 |
| `run_yolo_inference()` | L219-234 | 线程安全的 YOLO 推理封装 |
| `draw_defects()` | L240-283 | 绘制检测结果并按日期/样品编号保存 |

### 2. `make_app()` — 应用工厂

创建 Tornado Application 实例，注册路由和处理器参数。

---

## 数据流

### 请求格式

```json
{
  "job_id": "任务ID",
  "sample_id": "样品编号",
  "pose_id": "姿态/位置ID",
  "file_names": ["image.jpg"],
  "relative_dir": "images/"
}
```

### 成功响应

```json
{
  "error_code": 0,
  "error_msg": "OK",
  "data": {
    "product_type": "",
    "job_id": "任务ID",
    "pose_id": "姿态ID",
    "results": [
      {
        "code": "CuoLouZhuang",
        "box": [x1, y1, x2, y2],
        "area": 10000,
        "length": 100,
        "confidence": 0.12
      }
    ],
    "file_names": ["image.jpg"]
  }
}
```

### 错误响应

```json
{
  "error_code": 999,
  "error_msg": "False"
}
```

---

## 配置文件

文件路径：`config/assemble_detect_item.json`

结构示例：

```json
{
  "pose_id_1": {
    "A区域": {
      "region": [x1, y1, x2, y2],
      "threshold": 0.25
    },
    "B区域": {
      "region": [x1, y1, x2, y2],
      "threshold": 0.3
    }
  }
}
```

| 字段 | 说明 |
|------|------|
| `pose_id` | 姿态/工位标识，对应不同的检测区域配置 |
| `region` | 裁剪区域坐标 `[x1, y1, x2, y2]` |
| `threshold` | 置信度阈值，超过此值视为检测到缺陷 |

---

## 检测逻辑

服务采用**反向检测**逻辑：

1. 对每个配置的 `region` 裁剪 ROI
2. YOLO 推理，获取所有检测框的置信度
3. **若存在置信度 > threshold 的检测结果** → 该区域确认有缺陷，跳过（不返回）
4. **若无高置信度检测** → 返回该区域作为疑似缺陷区域（`code: CuoLouZhuang`）

> 即：返回的是"未检测到明确缺陷但可能存在问题"的区域。

---

## 结果保存

检测结果图像异步保存到：

```
res/
  └── {YYYY}/
      └── {MM}/
          └── {DD}/
              └── {sample_id}/
                  └── detected_{原文件名}.jpg
```

---

## 依赖

| 依赖 | 用途 |
|------|------|
| `tornado` | Web 框架，提供异步 HTTP 服务 |
| `ultralytics` | YOLO 目标检测模型 |
| `opencv-python (cv2)` | 图像读取、裁剪、绘制 |
| `numpy` | 数值计算 |
| `loguru` | 日志管理（按日轮转，保留7天） |

---

## 运行

```bash
python detect_server.py
```

服务启动后监听 `http://0.0.0.0:8002`

---

## 关键设计

| 设计点 | 说明 |
|--------|------|
| **异步非阻塞** | 模型推理和图像绘制通过 `ThreadPoolExecutor(4)` + `run_in_executor` 异步执行 |
| **线程安全** | `threading.Lock` 保护 YOLO 模型推理，避免并发竞争 |
| **日志轮转** | loguru 按日切分，保留 7 天，旧日志自动 zip 压缩 |
| **Ultralytics 隔离** | 设置 `ULTRALYTICS_SETTINGS` 和 `ULTRALYTICS_CACHE` 到项目本地目录，避免权限问题 |
