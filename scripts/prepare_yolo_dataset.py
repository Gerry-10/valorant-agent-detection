"""
Convert Pascal VOC XML to YOLO dataset for Ultralytics YOLOv8 training.
将 Pascal VOC XML 转为 YOLO 数据集，供 YOLOv8 训练。

Default paths (repo root) / 默认路径:
  images: data/train/images/*.jpg  |  annotations: data/train/annotations/*.xml
Outputs / 输出:
  yolo_dataset/images/{train,val}, labels/{train,val}, data.yaml, classes.txt

Usage / 用法:
  py scripts/prepare_yolo_dataset.py
  py scripts/prepare_yolo_dataset.py --val-ratio 0.2 --copy
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = PROJECT_ROOT / "data" / "train" / "images"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data" / "train" / "annotations"
DEFAULT_OUT = PROJECT_ROOT / "yolo_dataset"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pascal VOC -> YOLO dataset + data.yaml")
    p.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="训练图片目录")
    p.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS, help="VOC XML 目录")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出根目录")
    p.add_argument("--val-ratio", type=float, default=0.2, help="验证集占比")
    p.add_argument("--seed", type=int, default=42, help="随机划分种子")
    p.add_argument("--copy", action="store_true", help="始终复制图片（默认先尝试硬链接）")
    return p.parse_args()


def collect_classes(ann_dir: Path) -> list[str]:
    names: set[str] = set()
    for xml_path in ann_dir.glob("*.xml"):
        try:
            root = ET.parse(xml_path).getroot()
            for obj in root.findall("object"):
                ne = obj.find("name")
                if ne is not None and ne.text:
                    names.add(ne.text.strip())
        except ET.ParseError:
            continue
    return sorted(names)


def voc_to_yolo_line(
    cls_id: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    iw: int,
    ih: int,
) -> str | None:
    xmin = max(0.0, min(float(iw - 1), xmin))
    xmax = max(0.0, min(float(iw - 1), xmax))
    ymin = max(0.0, min(float(ih - 1), ymin))
    ymax = max(0.0, min(float(ih - 1), ymax))
    if xmax <= xmin or ymax <= ymin:
        return None
    w_box = xmax - xmin
    h_box = ymax - ymin
    if w_box < 1 or h_box < 1:
        return None
    xc = (xmin + xmax) / 2.0 / iw
    yc = (ymin + ymax) / 2.0 / ih
    wn = w_box / iw
    hn = h_box / ih
    if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
        return None
    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def parse_xml_boxes(
    xml_path: Path, name_to_id: dict[str, int]
) -> list[tuple[int, float, float, float, float]]:
    root = ET.parse(xml_path).getroot()
    out: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        ne = obj.find("name")
        box = obj.find("bndbox")
        if ne is None or ne.text is None or box is None:
            continue
        raw = ne.text.strip()
        if raw not in name_to_id:
            continue
        cid = name_to_id[raw]
        try:
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
        except ValueError:
            continue
        out.append((cid, xmin, ymin, xmax, ymax))
    return out


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"):
        p = images_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def link_or_copy(src: Path, dst: Path, force_copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if force_copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_split(
    pairs: list[tuple[Path, Path]],
    img_out: Path,
    lbl_out: Path,
    name_to_id: dict[str, int],
    force_copy: bool,
) -> tuple[int, int]:
    """返回 (成功写入的图片数, 跳过数)。"""
    n_ok = 0
    n_skip = 0
    for img_path, xml_path in tqdm(pairs, desc=f"{img_out.parent.name}/{img_out.name}"):
        im = cv2.imread(str(img_path))
        if im is None:
            n_skip += 1
            continue
        ih, iw = im.shape[:2]
        lines: list[str] = []
        for cid, xmin, ymin, xmax, ymax in parse_xml_boxes(xml_path, name_to_id):
            s = voc_to_yolo_line(cid, xmin, ymin, xmax, ymax, iw, ih)
            if s:
                lines.append(s)
        if not lines:
            n_skip += 1
            continue
        stem = img_path.stem
        link_or_copy(img_path, img_out / img_path.name, force_copy)
        lbl_out.mkdir(parents=True, exist_ok=True)
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_ok += 1
    return n_ok, n_skip


def split_train_val(
    pairs: list[tuple[Path, Path]], val_ratio: float, seed: int
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    if n == 1:
        return shuffled, list(shuffled)
    n_val = int(n * val_ratio)
    n_val = max(1, n_val)
    n_val = min(n_val, n - 1)
    return shuffled[n_val:], shuffled[:n_val]


def write_yaml(out_root: Path, class_names: list[str]) -> Path:
    yaml_path = out_root / "data.yaml"
    tr = (out_root / "images" / "train").as_posix()
    va = (out_root / "images" / "val").as_posix()
    names_lines = "\n".join(f"  {i}: {n!r}" for i, n in enumerate(class_names))
    content = (
        f"path: {out_root.as_posix()}\n"
        f"train: {tr}\n"
        f"val: {va}\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_lines}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    images_dir: Path = args.images
    ann_dir: Path = args.annotations
    out_root: Path = args.out

    if not images_dir.is_dir():
        print(f"错误：图片目录不存在：{images_dir}", file=sys.stderr)
        sys.exit(1)
    if not ann_dir.is_dir():
        print(f"错误：标注目录不存在：{ann_dir}", file=sys.stderr)
        sys.exit(1)
    if not 0 < args.val_ratio < 1:
        print("错误：--val-ratio 应在 (0, 1) 之间。", file=sys.stderr)
        sys.exit(1)

    class_names = collect_classes(ann_dir)
    if not class_names:
        print("错误：未从 XML 解析到任何类别。", file=sys.stderr)
        sys.exit(1)
    name_to_id = {n: i for i, n in enumerate(class_names)}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    pairs: list[tuple[Path, Path]] = []
    missing_img = 0
    for xml_path in sorted(ann_dir.glob("*.xml")):
        img_path = find_image(images_dir, xml_path.stem)
        if img_path is None:
            missing_img += 1
            continue
        pairs.append((img_path, xml_path))

    if not pairs:
        print("错误：没有「XML + 同名图片」成对样本。", file=sys.stderr)
        sys.exit(1)

    train_pairs, val_pairs = split_train_val(pairs, args.val_ratio, args.seed)

    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lbl_train = out_root / "labels" / "train"
    lbl_val = out_root / "labels" / "val"

    for d in (img_train, img_val, lbl_train, lbl_val):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    nt, st = write_split(train_pairs, img_train, lbl_train, name_to_id, args.copy)
    nv, sv = write_split(val_pairs, img_val, lbl_val, name_to_id, args.copy)

    yaml_path = write_yaml(out_root, class_names)

    print()
    print("预处理完成。")
    print(f"  类别数 nc={len(class_names)}，列表见 {out_root / 'classes.txt'}")
    print(f"  XML 总数: {len(list(ann_dir.glob('*.xml')))}，有对应图片的配对: {len(pairs)}")
    if missing_img:
        print(f"  仅有 XML 无图片（已跳过）: {missing_img}")
    print(f"  训练集写入: {nt} 张（跳过 {st}）")
    print(f"  验证集写入: {nv} 张（跳过 {sv}）")
    print(f"  data.yaml: {yaml_path}")
    print()
    print("下一步训练:")
    print("  py scripts/train_yolo.py")


if __name__ == "__main__":
    main()
