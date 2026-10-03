#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""key_scene_editor.py — 视频片段勾选拼接前端（每段加淡入淡出）。

功能：
  1. 给定一个目录，列出该目录下所有视频（mp4/mkv/mov/avi/webm 等），每个视频旁一个
     复选框，默认全部勾选；
  2. 页面上输入目录路径，输入框右侧的「拼接选中视频」按钮一旦按下，就按列表顺序把
     勾选的视频拼接成一段，且每段前后各加一次 fade in / fade out（默认 0.5 秒，可在
     页面上调整）；
  3. 拼接结果写到 <目录>/concat_fade.mp4（列表时自动排除该文件）。

实现要点：
  - 对每段先 re-encode：fade 淡入淡出 + 统一分辨率/帧率/pix_fmt/音频格式（AAC 48k 立体声），
    再用 ffmpeg concat demuxer 无损拼接（-c copy）。分辨率/帧率取第一个视频的规格，其余
    视频等比缩放 + 黑边补齐到相同尺寸。
  - 音频：所有勾选视频都有音轨时保留音轨；否则丢弃音轨（输出纯视频）。

用法：
  python key_scene_editor.py [目录] [--host 127.0.0.1] [--port 7100]

依赖：
  flask、ffmpeg / ffprobe（在 PATH 中，或通过 FFMPEG / FFPROBE 环境变量指定）。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Windows 控制台统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from flask import Flask, jsonify, request, send_from_directory

from ltx_common import FFMPEG, FFPROBE, concat_videos

SCRIPT_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=None)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts"}
FADE_DURATION = 0.5             # 默认每段淡入/淡出时长（秒）
OUTPUT_NAME = "concat_fade.mp4"  # 拼接结果文件名（列表时会被排除）

DEFAULT_DIR = ""  # 命令行传入的默认目录，供前端预填


# --------------------------------------------------------------------------- #
# 目录列举 / 自然排序
# --------------------------------------------------------------------------- #
def _natural_key(name):
    """自然排序键：按数字大小而非字典序，如 output_kf2 排在 output_kf10 前。"""
    key = []
    for p in re.split(r"(\d+)", str(name)):
        key.append((1, int(p)) if p.isdigit() else (0, p.lower()))
    return key


def list_videos(directory):
    """列出目录下所有视频文件名（自然排序），排除拼接结果文件。"""
    d = Path(directory)
    if not d.is_dir():
        raise ValueError(f"目录不存在：{d}")
    names = []
    for f in d.iterdir():
        if not f.is_file() or f.name == OUTPUT_NAME:
            continue
        if f.suffix.lower() in VIDEO_EXTS:
            names.append(f.name)
    names.sort(key=_natural_key)
    return names


# --------------------------------------------------------------------------- #
# 探测 / 淡入淡出 / 拼接
# --------------------------------------------------------------------------- #
def _probe(path):
    """用 ffprobe 读取视频的时长/分辨率/帧率/是否有音轨。"""
    cmd = [FFPROBE, "-v", "error", "-show_streams", "-show_format",
           "-of", "json", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe 读取失败：{Path(path).name}\n{r.stderr.strip()}")
    info = json.loads(r.stdout or "{}")
    streams = info.get("streams") or []
    fmt = info.get("format") or {}

    duration = None
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        duration = None

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise RuntimeError(f"不是视频文件（无视频流）：{Path(path).name}")

    width = video.get("width")
    height = video.get("height")
    fps = None
    for field in ("r_frame_rate", "avg_frame_rate"):
        raw = str(video.get(field) or "")
        if "/" in raw:
            num, den = raw.split("/", 1)
            try:
                den = float(den)
                if den:
                    fps = float(num) / den
                    break
            except (ValueError, ZeroDivisionError):
                continue

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {
        "duration": duration,
        "width": int(width) if width else None,
        "height": int(height) if height else None,
        "fps": fps or 24.0,
        "has_audio": has_audio,
    }


def _fade_segment(inp, outp, meta, fade_in, fade_out,
                  target_w, target_h, target_fps, keep_audio):
    """对单个视频加淡入淡出并统一规格，输出到 outp。"""
    dur = meta["duration"] or 0.0
    st_out = max(0.0, dur - fade_out)
    vf = (
        f"fade=t=in:st=0:d={fade_in:.3f},"
        f"fade=t=out:st={st_out:.3f}:d={fade_out:.3f},"
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={target_fps:g}"
    )
    cmd = [
        FFMPEG, "-y", "-i", str(inp),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
    ]
    if keep_audio and meta["has_audio"]:
        cmd += ["-c:a", "aac", "-ar", "48000", "-ac", "2"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", str(outp)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"淡入淡出处理失败：{Path(inp).name}\n{r.stderr.strip()[-1500:]}")
    return outp


def concat_with_fades(directory, names, fade=FADE_DURATION):
    """把勾选的视频（按顺序）加淡入淡出后拼接，返回输出文件路径。"""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"目录不存在：{directory}")
    if not names:
        raise ValueError("没有勾选任何视频")

    paths = []
    for n in names:
        p = directory / n
        if not p.is_file():
            raise ValueError(f"视频不存在：{n}")
        paths.append(p)

    metas = [_probe(p) for p in paths]
    first = metas[0]
    tw = first["width"] or 1280
    th = first["height"] or 720
    tfps = first["fps"] or 24.0
    keep_audio = all(m["has_audio"] for m in metas)

    tmp_dir = Path(tempfile.mkdtemp(prefix="key_scene_editor_"))
    seg_paths = []
    try:
        for i, (p, m) in enumerate(zip(paths, metas)):
            dur = m["duration"] or 0.0
            f = min(fade, dur / 2.0) if dur > 0 else 0.0
            seg = tmp_dir / f"seg_{i:04d}.mp4"
            _fade_segment(p, seg, m, f, f, tw, th, tfps, keep_audio)
            seg_paths.append(seg)
        outp = directory / OUTPUT_NAME
        concat_videos(seg_paths, outp, keep_audio=keep_audio)
        return str(outp)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 前端与 API
# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(str(SCRIPT_DIR), "key_scene_editor.html")


@app.route("/api/config")
def api_config():
    return jsonify({"ok": True, "default_dir": DEFAULT_DIR})


@app.route("/api/list", methods=["POST"])
def api_list():
    payload = request.get_json(force=True, silent=True) or {}
    directory = (payload.get("path") or "").strip()
    if not directory:
        raise ValueError("目录路径不能为空")
    names = list_videos(directory)
    items = []
    for n in names:
        dur = None
        try:
            dur = _probe(Path(directory) / n)["duration"]
        except Exception:
            dur = None
        items.append({"name": n, "duration": dur})
    return jsonify({
        "ok": True,
        "directory": str(Path(directory).resolve()),
        "videos": items,
    })


@app.route("/api/concat", methods=["POST"])
def api_concat():
    payload = request.get_json(force=True, silent=True) or {}
    directory = (payload.get("path") or "").strip()
    names = payload.get("videos") or []
    if not directory:
        raise ValueError("目录路径不能为空")
    try:
        fade = float(payload.get("fade", FADE_DURATION))
    except (TypeError, ValueError):
        fade = FADE_DURATION
    fade = max(0.0, min(5.0, fade))
    outp = concat_with_fades(directory, names, fade=fade)
    return jsonify({"ok": True, "output": outp})


@app.errorhandler(ValueError)
def _handle_value_error(e):
    return jsonify({"ok": False, "error": str(e)}), 400


@app.errorhandler(Exception)
def _handle_generic_error(e):
    return jsonify({"ok": False, "error": str(e)}), 500


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main():
    global DEFAULT_DIR
    parser = argparse.ArgumentParser(description="视频片段勾选拼接前端（每段加淡入淡出）")
    parser.add_argument("directory", nargs="?", default="",
                        help="默认目录（视频所在目录，可在页面里再改）")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=7100, help="监听端口（默认 7100）")
    args = parser.parse_args()

    DEFAULT_DIR = args.directory.strip()
    print(f"请在浏览器打开 http://{args.host}:{args.port}")
    if DEFAULT_DIR:
        print(f"默认目录：{DEFAULT_DIR}")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
