/**
 * Valorant detection UI — calls FastAPI /api/predict/*
 * 前端调用本地推理 API
 */
(function () {
  const apiBase = "";

  const apiEl = document.getElementById("api-base");
  if (apiEl) {
    apiEl.textContent = window.location.origin || "http://127.0.0.1:8000";
  }

  const confSlider = document.getElementById("conf-slider");
  const confVal = document.getElementById("conf-val");

  function getConf() {
    const v = parseFloat(confSlider.value, 10);
    return Number.isFinite(v) ? v : 0.5;
  }

  confSlider.addEventListener("input", () => {
    confVal.textContent = getConf().toFixed(2);
  });

  const imgInput = document.getElementById("img-input");
  const imgBtn = document.getElementById("img-btn");
  const imgStatus = document.getElementById("img-status");
  const imgMeta = document.getElementById("img-meta");
  const imgOut = document.getElementById("img-out");
  const imgFilename = document.getElementById("img-filename");

  const vidInput = document.getElementById("vid-input");
  const vidBtn = document.getElementById("vid-btn");
  const vidStatus = document.getElementById("vid-status");
  const vidMeta = document.getElementById("vid-meta");
  const vidOut = document.getElementById("vid-out");
  const vidFilename = document.getElementById("vid-filename");

  vidOut.addEventListener("error", function () {
    vidStatus.textContent +=
      " 若无法播放 / If playback fails：安装 ffmpeg 并重启服务，或下载后用 VLC。Install ffmpeg & restart, or use VLC.";
  });

  imgInput.addEventListener("change", () => {
    imgBtn.disabled = !imgInput.files?.length;
    imgFilename.textContent = imgInput.files?.[0]?.name || "";
  });

  vidInput.addEventListener("change", () => {
    vidBtn.disabled = !vidInput.files?.length;
    vidFilename.textContent = vidInput.files?.[0]?.name || "";
  });

  imgBtn.addEventListener("click", async () => {
    const f = imgInput.files?.[0];
    if (!f) return;
    imgStatus.textContent =
      "上传并推理中… / Uploading & inferring…";
    imgMeta.textContent = "";
    imgOut.hidden = true;

    const fd = new FormData();
    fd.append("file", f);
    const conf = getConf();
    const q = new URLSearchParams({ conf: String(conf), imgsz: "640" });

    try {
      const res = await fetch(apiBase + "/api/predict/image?" + q.toString(), {
        method: "POST",
        body: fd,
      });
      const text = await res.text();
      if (!res.ok) {
        imgStatus.textContent = "失败 / Error: " + text;
        return;
      }
      const data = JSON.parse(text);
      const th = (data.conf_threshold ?? conf).toFixed(2);
      imgStatus.textContent =
        "完成：共 " +
        (data.count ?? 0) +
        " 个目标 · Done: " +
        (data.count ?? 0) +
        " object(s). 阈值 threshold " +
        th +
        ".";
      imgMeta.textContent = JSON.stringify(data.detections, null, 2);
      if (data.image_base64_jpeg) {
        imgOut.src =
          "data:" + (data.mime || "image/jpeg") + ";base64," + data.image_base64_jpeg;
        imgOut.hidden = false;
      }
    } catch (e) {
      imgStatus.textContent =
        "请求异常 / Request error: " + (e && e.message ? e.message : e);
    }
  });

  vidBtn.addEventListener("click", async () => {
    const f = vidInput.files?.[0];
    if (!f) return;
    vidStatus.textContent =
      "上传并处理视频中（可能较久）… / Processing video (may take a while)…";
    vidMeta.textContent = "";
    vidOut.hidden = true;
    if (vidOut.src && vidOut.src.startsWith("blob:")) {
      URL.revokeObjectURL(vidOut.src);
    }

    const fd = new FormData();
    fd.append("file", f);
    const conf = getConf();
    const q = new URLSearchParams({ conf: String(conf), imgsz: "640" });

    try {
      const t0 = performance.now();
      const res = await fetch(apiBase + "/api/predict/video?" + q.toString(), {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        const errText = await res.text();
        vidStatus.textContent = "失败 / Error: " + errText;
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      vidOut.src = url;
      vidOut.hidden = false;
      const sec = ((performance.now() - t0) / 1000).toFixed(1);
      const serverTime = res.headers.get("X-Process-Time-Sec");
      const enc = res.headers.get("X-Video-Encode");
      const confHdr = res.headers.get("X-Conf-Threshold");
      vidStatus.textContent =
        "完成 / Done · 浏览器 ~" +
        sec +
        "s · 服务端 server ~" +
        (serverTime || "?") +
        "s" +
        (enc ? " · 编码 encode " + enc : "") +
        " · conf " +
        (confHdr || conf.toFixed(2)) +
        ". 可右键另存 / Right-click to save.";
      vidMeta.textContent =
        "Content-Type: " + (blob.type || "video/mp4");
    } catch (e) {
      vidStatus.textContent =
        "请求异常 / Request error: " + (e && e.message ? e.message : e);
    }
  });
})();
