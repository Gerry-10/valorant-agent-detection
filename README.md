# Valorant Agent Detection · 无畏契约特工目标检测

**Author / 作者:** [Gerry-10](https://github.com/Gerry-10)  
**Repo / 仓库:** `https://github.com/Gerry-10/valorant-agent-detection`（创建后使用下方 push 命令绑定）

**Bilingual README / 双语说明** — see **§ Chinese** and **§ English** below.

Personal CV project: **Pascal VOC → YOLOv8** pipeline, local training, and a small **FastAPI + web UI** for image/video inference.  
个人计算机视觉项目：**Pascal VOC 转 YOLOv8** 数据流水线、本地训练，以及 **FastAPI + 网页** 的图片/视频推理演示。

> **Disclaimer / 声明**  
> Riot Games trademarks belong to their owners. This repo is for **learning only**, not affiliated with or endorsed by Riot.  
> 瓦罗兰特相关商标归权利人所有。本仓库仅用于**学习**，与 Riot 无关。

---

## GitHub: create repo · 创建仓库时怎么填

按你截图中的 **Create a new repository** 页面，建议如下（本地已有 `README` / `.gitignore` / `LICENSE`，**不要**在网页上再生成一份，避免冲突）：

| 项 | 建议 |
|----|------|
| **Owner** | `Gerry-10` ✓ |
| **Repository name** | `valorant-agent-detection` ✓ |
| **Description** | 可粘贴下面一行（中英，约 200 字以内）：<br>`YOLOv8 agent detection in Valorant screenshots — VOC→YOLO pipeline, FastAPI & bilingual web UI. Learning / CV demo.` <br>中文可写：`基于 YOLOv8 的游戏画面特工检测，含 VOC 转 YOLO、FastAPI 与双语网页演示。`（若字数超限可删一句） |
| **Public** | **Public**（简历展示）或 **Private**（仅自己可见） |
| **Add README** | **关闭 Off**（你仓库里已有 `README.md`） |
| **Add .gitignore** | **No .gitignore**（你已有自定义 `.gitignore`，勿选 Python 模板覆盖） |
| **Add license** | **No license**（你已有根目录 `LICENSE` 文件） |

创建完成后，在**本地项目根目录**执行（把远程绑到你的仓库）：

```bash
git remote add origin https://github.com/Gerry-10/valorant-agent-detection.git
git branch -M main
git push -u origin main
```

若已存在 `origin`，改用：`git remote set-url origin https://github.com/Gerry-10/valorant-agent-detection.git`

---

## What is .gitignore? · `.gitignore` 是什么？要上传数据和训练结果吗？

**`.gitignore`** 是一个文本列表，告诉 Git：**这些路径/文件不要跟踪、不要提交**。适合忽略体积大、隐私、或每人本机不同的内容。

本仓库已配置忽略（**不应**推到 GitHub）：

| 类型 | 路径示例 | 原因 |
|------|----------|------|
| 原始数据与标注 | `data/` | 体积大、可能含隐私；别人按 `docs/DATA_LAYOUT.md` 自备数据 |
| 转换后的 YOLO 数据 | `yolo_dataset/` | 可由脚本从 VOC 再生 |
| 训练运行目录 | `runs/`（含 `results.csv`、曲线图、`weights/` 等） | 体积大；`*.pt` 也已忽略 |
| 权重文件 | `*.pt` | 常数百 MB，且与机器/训练轮次绑定 |
| Web 推理缓存 | `web/outputs/` | 本地临时视频/中间文件 |

**结论：** 代码、文档、脚本 **上传**；数据、训练产物、权重 **默认不上传**。若你希望公开**某次训练的 mAP 截图**作展示，可单独导出小图放进 `docs/`（注意别带隐私），再 `git add`——这是例外操作。

---

## Chinese · 中文说明

### 功能概览

| 模块 | 说明 |
|------|------|
| `scripts/prepare_yolo_dataset.py` | 将 `data/train` 下 VOC XML 转为 YOLO 格式，划分 train/val，生成 `yolo_dataset/data.yaml` |
| `scripts/train_yolo.py` | 使用 Ultralytics YOLOv8 微调（底层 PyTorch） |
| `web/app.py` | FastAPI：图片/视频推理，置信度阈值可调，视频可选 ffmpeg 转 H.264 便于浏览器播放 |

### 目录结构（提交到 Git 的部分）

```text
valorant_cv_project/
├── README.md                 # 本文件（中英）
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

以下目录**默认不提交**（见 `.gitignore`）：`data/`、`yolo_dataset/`、`runs/`、`*.pt`、`web/outputs/`。

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

See tree above. **Not committed** (`.gitignore`): `data/`, `yolo_dataset/`, `runs/`, `*.pt`, `web/outputs/`.

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

See [LICENSE](LICENSE) — Copyright **Gerry-10** (2025). Use a real legal name instead of the GitHub handle if you prefer.  
许可证见 `LICENSE`，著作权署名为 **Gerry-10**；若需法律效力可改为真实姓名。

### Web UI fonts / 网页字体

The demo page loads **Plus Jakarta Sans** from Google Fonts (requires internet on first visit).  
演示页使用 Google Fonts 的 Plus Jakarta Sans（首次打开需联网加载字体）。

---

## Manual checklist for Gerry-10 · 仍需你本人操作的事项

1. **GitHub 网页**：按上文「创建仓库」表格创建仓库；**Description** 可自选是否填写。  
2. **首次 push**：本地执行 `git commit`（若尚未提交）→ `git remote add` → `git push`；按提示完成 GitHub 登录或 Token。  
3. **仓库主页**：在 GitHub 上为仓库添加 **Topics**（如 `yolov8`, `pytorch`, `fastapi`, `computer-vision`）便于被搜索。  
4. **LICENSE**：若希望署名为**真实姓名**而非 GitHub ID，请编辑 `LICENSE` 第一行 `Copyright`。  
5. **README 里的仓库链接**：若你最终仓库名**不是** `valorant-agent-detection`，请全文替换 README 顶部的 URL。  
6. **可选**：在 README 加一张 Web 界面或检测效果图（放到 `docs/images/` 并缩小体积），增强展示效果。
