"""
Valorant agent detection — YOLOv8 training entrypoint (Ultralytics, PyTorch backend).
特工检测 — YOLOv8 训练入口（Ultralytics，底层 PyTorch）。

Uses yolo_dataset/data.yaml by default (run scripts/prepare_yolo_dataset.py first).
默认读取 yolo_dataset/data.yaml（需先运行预处理脚本）。

GPU tip (e.g. RTX 4070 Super ~12GB): install CUDA PyTorch from pytorch.org, then ultralytics.
Lower --batch if OOM. / 先装 CUDA 版 PyTorch，再 ultralytics；显存不足则减小 batch。

  py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  py -m pip install ultralytics

  py scripts/train_yolo.py
  py scripts/train_yolo.py --model yolov8m.pt --batch 16 --epochs 150
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "yolo_dataset" / "data.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "runs" / "valorant"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8 on Valorant yolo_dataset")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="data.yaml 路径")
    p.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="预训练权重：yolov8n/s/m/l/x.pt",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--batch",
        type=int,
        default=24,
        help="批大小；显存不足时减小",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help='设备：默认自动用 GPU 0；可填 "cpu" 或 "0","1"',
    )
    p.add_argument("--workers", type=int, default=8, help="DataLoader 线程数，Windows 异常时可改为 4 或 0")
    p.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="训练输出根目录（默认项目下 runs/valorant）")
    p.add_argument("--name", type=str, default="train", help="本次运行子目录名")
    p.add_argument("--patience", type=int, default=50, help="早停：验证集若干 epoch 无提升则停止，0 关闭")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        print(f"错误：找不到 data.yaml：{args.data}", file=sys.stderr)
        print("请先运行：py scripts/prepare_yolo_dataset.py", file=sys.stderr)
        sys.exit(1)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print("错误：未安装 ultralytics 或 PyTorch。", file=sys.stderr)
        print("请执行：py -m pip install ultralytics", file=sys.stderr)
        print("并安装 CUDA 版 PyTorch：https://pytorch.org", file=sys.stderr)
        raise SystemExit(1) from e

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        device = 0
        print(f"使用 GPU：{torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("警告：未检测到 CUDA，将使用 CPU（会很慢）。请安装 CUDA 版 PyTorch。")

    project_dir = args.project.resolve()
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        patience=args.patience if args.patience > 0 else 0,
        seed=args.seed,
        exist_ok=True,
    )

    print()
    weights_dir = project_dir / args.name / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    print(f"训练结束。权重目录：{weights_dir}")
    if best.is_file():
        print(f"  best.pt : {best.resolve()}")
    if last.is_file():
        print(f"  last.pt : {last.resolve()}")


if __name__ == "__main__":
    main()
