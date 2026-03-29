"""
Valorant agent detection — FastAPI inference API + static demo UI.
无畏契约特工检测 — FastAPI 推理服务与静态演示页。

Run from repo root / 在项目根目录启动:
  py -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
Browser / 浏览器: http://127.0.0.1:8000

Env / 环境变量:
  YOLO_WEIGHTS — optional path to weights (.pt) / 自定义权重路径
  FFMPEG_PATH, YOLO_FFMPEG — optional path to ffmpeg.exe / ffmpeg 路径

Video: Ultralytics on Windows often writes .avi; browsers may not play it.
We locate ffmpeg (including WinGet package dir if PATH is stale) and transcode to H.264 MP4;
fallback OpenCV remux. / 自动查找 ffmpeg 转 H.264；否则 OpenCV 回退。
"""

from __future__ import annotations

import base64
import mimetypes
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "valorant" / "train" / "weights" / "best.pt"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = Path(os.environ.get("YOLO_WEIGHTS", str(DEFAULT_WEIGHTS))).resolve()

app = FastAPI(title="Valorant Agent Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None


def _which_ffmpeg() -> str | None:
    """解析 ffmpeg 路径：PATH → 环境变量 → WinGet 典型目录（避免终端未刷新 PATH）。"""
    env = os.environ.get("FFMPEG_PATH") or os.environ.get("YOLO_FFMPEG")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        if p.is_dir():
            exe = p / "ffmpeg.exe"
            if exe.is_file():
                return str(exe)

    for name in ("ffmpeg", "ffmpeg.exe"):
        w = shutil.which(name)
        if w:
            return w

    local = os.environ.get("LOCALAPPDATA", "")
    if local:
        winget_pkg = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkg.is_dir():
            matches = list(winget_pkg.glob("Gyan.FFmpeg*/ffmpeg-*-full_build/bin/ffmpeg.exe"))
            if matches:
                matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(matches[0])

    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    for cand in (Path(pf) / "ffmpeg" / "bin" / "ffmpeg.exe",):
        if cand.is_file():
            return str(cand)

    return None


def _ffmpeg_to_h264_mp4(src: Path, dst: Path) -> bool:
    """将任意视频转为浏览器可播的 H.264 + yuv420p MP4（需系统安装 ffmpeg）。"""
    exe = _which_ffmpeg()
    if not exe:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=3600, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and dst.is_file() and dst.stat().st_size > 0


def _opencv_to_mp4v(src: Path, dst: Path) -> bool:
    """无 ffmpeg 时用 OpenCV 重封装为 mp4v（多数浏览器可播）。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return False
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        n += 1
    writer.release()
    cap.release()
    return n > 0 and dst.is_file() and dst.stat().st_size > 0


def ensure_browser_friendly_video(src: Path) -> tuple[Path, str]:
    """
    返回 (路径, media_type)。
    Ultralytics 在 Windows 上常输出 .avi（浏览器难播），尽量转为 H.264 MP4。
    """
    ext = src.suffix.lower()
    if ext == ".mp4":
        guessed, _ = mimetypes.guess_type(str(src))
        return src, guessed or "video/mp4"

    out_mp4 = src.with_name(src.stem + "_browser.mp4")
    if _ffmpeg_to_h264_mp4(src, out_mp4):
        return out_mp4, "video/mp4"
    out_mp4 = src.with_name(src.stem + "_opencv.mp4")
    if _opencv_to_mp4v(src, out_mp4):
        return out_mp4, "video/mp4"

    guessed, _ = mimetypes.guess_type(str(src))
    return src, guessed or ("video/x-msvideo" if ext == ".avi" else "video/mp4")


def get_model():
    global _model
    if _model is None:
        if not WEIGHTS_PATH.is_file():
            raise RuntimeError(
                f"未找到权重文件: {WEIGHTS_PATH}\n请先训练或设置环境变量 YOLO_WEIGHTS"
            )
        from ultralytics import YOLO

        _model = YOLO(str(WEIGHTS_PATH))
    return _model


@app.get("/health")
def health():
    ok = WEIGHTS_PATH.is_file()
    return {"status": "ok" if ok else "no_weights", "weights": str(WEIGHTS_PATH), "exists": ok}


@app.post("/api/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    imgsz: int = Query(640, ge=320, le=1280, description="输入边长"),
    conf: float = Query(
        0.5,
        ge=0.05,
        le=0.999,
        description="置信度阈值：低于此值的框不会输出（默认 0.5，可减少误检）",
    ),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图片文件（image/*）")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(400, "图片过大（>20MB）")

    try:
        model = get_model()
        arr = np.frombuffer(raw, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            raise ValueError("无法解码图片")
        results = model.predict(
            source=im,
            imgsz=imgsz,
            conf=conf,
            verbose=False,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"推理失败: {e}") from e

    r0 = results[0]
    detections = []
    if r0.boxes is not None and len(r0.boxes) > 0:
        boxes = r0.boxes
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            detections.append(
                {
                    "class_id": cid,
                    "name": model.names.get(cid, str(cid)),
                    "confidence": float(boxes.conf[i].item()),
                    "box_xyxy": [float(x) for x in boxes.xyxy[i].tolist()],
                }
            )

    plot_bgr = r0.plot()
    ok, buf = cv2.imencode(".jpg", plot_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(500, "编码结果图失败")
    image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    return {
        "detections": detections,
        "count": len(detections),
        "conf_threshold": conf,
        "image_base64_jpeg": image_b64,
        "mime": "image/jpeg",
    }


@app.post("/api/predict/video")
async def predict_video(
    file: UploadFile = File(...),
    imgsz: int = Query(640, ge=320, le=1280),
    conf: float = Query(
        0.5,
        ge=0.05,
        le=0.999,
        description="置信度阈值，与图片接口一致",
    ),
):
    fn = (file.filename or "").lower()
    if file.content_type and file.content_type.startswith("video/"):
        pass
    elif fn.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
        pass
    else:
        raise HTTPException(400, "请上传常见视频文件（如 .mp4）")

    raw = await file.read()
    if len(raw) > 500 * 1024 * 1024:
        raise HTTPException(400, "视频过大（>500MB），请压缩或分段处理")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    job = uuid.uuid4().hex[:12]
    tmp_path = Path(tempfile.gettempdir()) / f"valorant_vid_{job}{suffix}"
    out_name = f"vid_{job}"

    try:
        tmp_path.write_bytes(raw)
        model = get_model()
        t0 = time.perf_counter()
        for _ in model.predict(
            source=str(tmp_path),
            imgsz=imgsz,
            conf=conf,
            save=True,
            verbose=False,
            stream=True,
            project=str(OUTPUT_DIR),
            name=out_name,
            exist_ok=True,
        ):
            pass
        elapsed = time.perf_counter() - t0
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"视频推理失败: {e}") from e
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass

    out_dir = OUTPUT_DIR / out_name
    if not out_dir.is_dir():
        raise HTTPException(500, "未生成输出目录")

    candidates = sorted(
        list(out_dir.glob("*.mp4"))
        + list(out_dir.glob("*.avi"))
        + list(out_dir.glob("*.mov")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise HTTPException(500, "未找到输出视频文件")

    out_file = candidates[0]
    playable, media_type = ensure_browser_friendly_video(out_file)
    extra = "ffmpeg" if playable.name.endswith("_browser.mp4") else (
        "opencv" if playable.name.endswith("_opencv.mp4") else "original"
    )
    return FileResponse(
        path=str(playable),
        media_type=media_type,
        filename=f"predicted_{playable.name}",
        headers={
            "X-Process-Time-Sec": f"{elapsed:.3f}",
            "X-Video-Encode": extra,
            "X-Conf-Threshold": f"{conf:.3f}",
        },
    )


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = STATIC_DIR / "index.html"
    if not index_file.is_file():
        return HTMLResponse("<h1>缺少 static/index.html</h1>", status_code=500)
    return HTMLResponse(index_file.read_text(encoding="utf-8"))
