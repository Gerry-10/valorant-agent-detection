# Valorant Agent Detection · 无畏契约特工目标检测

**Author / 作者:** [Gerry-10](https://github.com/Gerry-10)  
**Repository / 仓库:** https://github.com/Gerry-10/valorant-agent-detection

Personal CV project: **Pascal VOC → YOLOv8** pipeline, local training, and a small **FastAPI + web UI** for image/video inference.  
个人计算机视觉项目：**Pascal VOC 转 YOLOv8** 数据流水线、本地训练，以及 **FastAPI + 网页** 的图片/视频推理演示。

README below is **bilingual** — **Chinese** first, then **English**.

> **Disclaimer / 声明**  
> Riot Games trademarks belong to their owners. This repo is for **learning only**, not affiliated with or endorsed by Riot.  
> 瓦罗兰特相关商标归权利人所有。本仓库仅用于**学习**，与 Riot 无关。

---

## Chinese · 中文说明

### 功能概览

| 模块 | 说明 |
|------|------|
| `scripts/prepare_yolo_dataset.py` | 将 `data/train` 下 VOC XML 转为 YOLO 格式，划分 train/val，生成 `yolo_dataset/data.yaml` |
| `scripts/train_yolo.py` | 使用 Ultralytics YOLOv8 微调（底层 PyTorch） |
| `web/app.py` | FastAPI：图片/视频推理，置信度阈值可调，视频可选 ffmpeg 转 H.264 便于浏览器播放 |

### 目录结构（纳入版本库的部分）

```text
valorant_cv_project/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   └── DATA_LAYOUT.md        # 数据目录约定（中英）
├── scripts/
│   ├── prepare_yolo_dataset.py
│   └── train_yolo.py
└── web/
    ├── app.py
    └── static/               # 前端静态文件
```

**未纳入版本库的内容**（见根目录 `.gitignore`）：原始数据与标注 `data/`、转换后的 `yolo_dataset/`、训练输出 `runs/`、权重 `*.pt`、Web 临时输出 `web/outputs/` 等，体积大或可由脚本再生，请在本机自行准备。

### 环境依赖

1. **Python 3.10+**（推荐 3.12）
2. **CUDA 版 PyTorch**（GPU 训练/推理）：从 [pytorch.org](https://pytorch.org) 选择 Windows + CUDA，例如：
   ```bash
   py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
3. 安装其余依赖：
   ```bash
   py -m pip install -r requirements.txt
   ```
4. **（可选）ffmpeg**：视频在浏览器内播放更稳定；Windows 可用 `winget install Gyan.FFmpeg`。未安装时服务会尝试 OpenCV 转封装。

### 数据放置

详见 [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md)。  
简要：将训练图片放在 `data/train/images/`，VOC XML 放在 `data/train/annotations/`，文件名（不含扩展名）一一对应。

### 命令速查

在项目根目录执行：

```bash
# 1) 数据预处理
py scripts/prepare_yolo_dataset.py --copy

# 2) 训练（需先有 yolo_dataset/data.yaml）
py scripts/train_yolo.py

# 3) 启动 Web（浏览器打开 http://127.0.0.1:8000）
py -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

环境变量（可选）：

- `YOLO_WEIGHTS`：自定义 `best.pt` 路径  
- `FFMPEG_PATH` / `YOLO_FFMPEG`：自定义 `ffmpeg.exe` 路径（PATH 未刷新时有用）

### API 摘要

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 静态演示页 |
| GET | `/health` | 检查权重文件是否存在 |
| POST | `/api/predict/image?conf=0.5&imgsz=640` | 上传图片，`conf` 为置信度阈值 |
| POST | `/api/predict/video?conf=0.5` | 上传视频，返回带检测框的视频文件 |

---

## English · 英文说明

### Overview

| Component | Description |
|-----------|-------------|
| `scripts/prepare_yolo_dataset.py` | Converts Pascal VOC XML under `data/train` to YOLO labels, train/val split, writes `yolo_dataset/data.yaml` |
| `scripts/train_yolo.py` | Fine-tunes Ultralytics YOLOv8 (PyTorch backend) |
| `web/app.py` | FastAPI inference for images/videos; tunable `conf`; optional ffmpeg for browser-friendly MP4 |

### Layout (tracked in Git)

Same tree as above. **Not committed** (see `.gitignore`): `data/`, `yolo_dataset/`, `runs/`, `*.pt`, `web/outputs/`, and other generated or large local files.

### Setup

1. Install **CUDA PyTorch** from [pytorch.org](https://pytorch.org) if you use GPU.  
2. `pip install -r requirements.txt`  
3. Optional: **ffmpeg** for reliable in-browser video playback.

### Data layout

See [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md).

### Quick commands

```bash
py scripts/prepare_yolo_dataset.py --copy
py scripts/train_yolo.py
py -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

Optional env: `YOLO_WEIGHTS`, `FFMPEG_PATH`, `YOLO_FFMPEG`.

### License

See [LICENSE](LICENSE) — Copyright **Gerry-10** (2025).

### Web UI fonts / 网页字体

The demo page loads **Plus Jakarta Sans** from Google Fonts (requires internet on first visit).  
演示页使用 Google Fonts 的 Plus Jakarta Sans（首次打开需联网加载字体）。
