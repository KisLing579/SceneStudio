#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""公共模块：被 generate_clips.py / concat_clips.py / concat_audio.py / add_audio.py 复用。

集中存放可配置常量、meta.json 读取、帧数换算、命令构建、抽帧、拼接与合成等工具函数。
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Windows 控制台默认编码可能是 cp1252/gbk，统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# --------------------------------------------------------------------------- #
# 可配置项
# --------------------------------------------------------------------------- #
FPS = 24                 # LTX 视频帧率
MAX_FRAMES = 121         # GPU 单次生成帧数上限
SEED = 42                # 固定种子（对应命令里的 --seed）
IMAGE_START = "0"        # --image 后的第一个位置参数
IMAGE_STRENGTH = "0.8"   # --image 后的第二个位置参数
TIMEOUT = 3600           # 单次生成超时（秒）

# LTX-2.5 模型路径（相对 LTX_WORKDIR，实际模型目录在工作目录的上一级）
MODELS = {
    "transformer": "../models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": "../models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": "../models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": "../models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors",
    "spatial_upsampler": "../models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

SCRIPT_DIR = Path(__file__).resolve().parent
WORKDIR = os.environ.get("LTX_WORKDIR") or str(SCRIPT_DIR)
CLIPS_DIR = SCRIPT_DIR / "clips"               # 分镜视频段目录
WORK_DIR = SCRIPT_DIR / "work"                 # 中间产物目录（续接帧、日志等）
RESULT_PATH = SCRIPT_DIR / "result.mp4"        # 结果视频
VIDEO_ONLY_PATH = WORK_DIR / "result_video.mp4"   # 纯视频结果（合成前）
AUDIO_CONCAT_PATH = WORK_DIR / "result_audio.m4a"  # 拼接后的配音


def _which_or_env(name, env_var):
    """优先使用环境变量指定的可执行文件，否则回退到 PATH 查找。"""
    exe = os.environ.get(env_var)
    if exe:
        return exe
    found = shutil.which(name)
    return found or name


FFMPEG = _which_or_env("ffmpeg", "FFMPEG")
FFPROBE = _which_or_env("ffprobe", "FFPROBE")


# --------------------------------------------------------------------------- #
# meta.json 读取
# --------------------------------------------------------------------------- #
def load_meta(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    task = data.get("task_data") or data
    prompts = task.get("prompts") or []
    images = task.get("images") or []
    durations = task.get("durations") or []
    audios = task.get("audios") or []
    lines = task.get("lines") or []
    if not prompts:
        raise RuntimeError("meta.json 中未找到 prompts 列表")
    if len(images) != len(prompts):
        raise RuntimeError(
            f"images 长度({len(images)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    if audios and len(audios) != len(prompts):
        raise RuntimeError(
            f"audios 长度({len(audios)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    if lines and len(lines) != len(prompts):
        raise RuntimeError(
            f"lines 长度({len(lines)}) 与 prompts 长度({len(prompts)}) 不一致"
        )
    return prompts, images, durations, audios, lines


def parse_shot_id(shot_id):
    """从 shot_id（如 scene3-shot2）解析出 (scene_num, shot_num)。"""
    parts = str(shot_id or "").split("-")

    def _num(s):
        m = re.search(r"\d+", s or "")
        return int(m.group()) if m else None

    scene_num = _num(parts[0]) if parts else None
    shot_num = _num(parts[1]) if len(parts) > 1 else None
    return scene_num, shot_num


def select_indices(prompts, scene=None, shot=None):
    """按场景 / 分镜过滤，返回要处理的分镜原始下标列表。"""
    if scene is None and shot is None:
        return list(range(len(prompts)))
    kept = []
    for idx, p in enumerate(prompts):
        scene_num, shot_num = parse_shot_id(p.get("shot_id"))
        if scene is not None and scene_num != scene:
            continue
        if shot is not None and shot_num != shot:
            continue
        kept.append(idx)
    if not kept:
        raise RuntimeError("没有匹配指定 --scene / --shot 的分镜")
    return kept


# --------------------------------------------------------------------------- #
# 帧数换算与切段
# --------------------------------------------------------------------------- #
def frame_segments(duration, fps=FPS, max_frames=MAX_FRAMES):
    """由秒数换算成若干段 num_frames 列表，每段均为 8n+1 且 <= max_frames。"""
    duration = float(duration)
    frames = 8 * round(duration * fps / 8) + 1
    frames = max(frames, 1)
    if frames <= max_frames:
        return [frames]
    payload = frames - 1
    cap = max_frames - 1
    segments = []
    while payload > 0:
        chunk = min(payload, cap)
        chunk -= chunk % 8
        segments.append(chunk + 1)
        payload -= chunk
    return segments


# --------------------------------------------------------------------------- #
# 命令构建与执行
# --------------------------------------------------------------------------- #
def build_command(prompt, image_path, frames, output_path):
    return [
        "uv", "run", "python", "-m", "ltx_pipelines.distilled",
        "--transformer-path", MODELS["transformer"],
        "--text-encoder-path", MODELS["text_encoder"],
        "--video-vae-path", MODELS["video_vae"],
        "--audio-vae-path", MODELS["audio_vae"],
        "--spatial-upsampler-path", MODELS["spatial_upsampler"],
        "--num-frames", str(frames),
        "--seed", str(SEED),
        "--output-path", str(output_path),
        "--prompt", prompt,
        "--image", str(image_path), IMAGE_START, IMAGE_STRENGTH,
    ]


def run_generation(command, log_path):
    """执行一条生成命令并等待结束；失败时抛出带日志尾部的异常。"""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as logf:
        logf.write((" ".join(command) + "\n").encode("utf-8"))
        logf.flush()
        try:
            proc = subprocess.run(
                command,
                cwd=WORKDIR,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"生成超时（>{TIMEOUT}s），日志见 {log_path}")
    if proc.returncode != 0:
        tail = _tail(log_path)
        raise RuntimeError(f"生成失败（返回码 {proc.returncode}），日志见 {log_path}\n{tail}")


def _tail(log_path, n=2000):
    try:
        return Path(log_path).read_text(encoding="utf-8", errors="replace")[-n:]
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# 最后一帧抽取（逻辑同 extract_last_frame.py）
# --------------------------------------------------------------------------- #
def extract_last_frame(video_path, image_path):
    """用 ffprobe 统计总帧数，ffmpeg 抽取最后一帧。"""
    video_path = str(video_path)
    image_path = str(image_path)
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)

    probe_cmd = [
        FFPROBE, "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    r = subprocess.run(probe_cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    value = r.stdout.strip()
    if not value.isdigit():
        raise RuntimeError(f"无法读取视频帧数: {r.stderr.strip() or value}")
    total = int(value)
    last_index = total - 1

    ff_cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{last_index})",
        "-frames:v", "1", "-y", image_path,
    ]
    r2 = subprocess.run(ff_cmd, capture_output=True, text=True,
                        encoding="utf-8", errors="replace")
    if r2.returncode != 0:
        raise RuntimeError(r2.stderr.strip())
    if not os.path.isfile(image_path):
        raise RuntimeError("抽帧失败: 未生成图像文件")
    return image_path


# --------------------------------------------------------------------------- #
# 拼接 / 合成（ffmpeg concat demuxer）
# --------------------------------------------------------------------------- #
def concat_videos(seg_paths, output_path, keep_audio=False):
    """按顺序无损拼接视频段（默认 -c:v copy -an 丢弃音频；keep_audio=True 时 -c copy 保留音轨）。"""
    output_path = str(output_path)
    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in seg_paths:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
    codec_args = ["-c", "copy"] if keep_audio else ["-c:v", "copy", "-an"]
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        *codec_args, output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if os.path.exists(list_file):
        os.remove(list_file)
    if r.returncode != 0:
        raise RuntimeError(f"合并视频失败: {r.stderr.strip()}")
    if not os.path.isfile(output_path):
        raise RuntimeError("合并视频失败: 未生成输出文件")
    return output_path


def concat_audios(audio_paths, output_path):
    """按顺序拼接多段配音为一段 AAC 音频。"""
    output_path = str(output_path)
    list_file = output_path + ".concat.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for p in audio_paths:
            f.write(f"file '{str(p).replace(chr(92), '/')}'\n")
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", list_file,
        "-c:a", "aac", "-b:a", "192k", output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if os.path.exists(list_file):
        os.remove(list_file)
    if r.returncode != 0:
        raise RuntimeError(f"拼接配音失败: {r.stderr.strip()}")
    if not os.path.isfile(output_path):
        raise RuntimeError("拼接配音失败: 未生成输出文件")
    return output_path


def mux_audio(video_path, audio_path, output_path):
    """把配音 mux 进视频（视频流无损拷贝，音频流 AAC）。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart",
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"合成配音失败: {r.stderr.strip()}")
    if not os.path.isfile(str(output_path)):
        raise RuntimeError("合成配音失败: 未生成输出文件")
    return str(output_path)


# --------------------------------------------------------------------------- #
# 中→英翻译（DeepSeek chat completions，与 Mindawaker text_engine 一致）
# --------------------------------------------------------------------------- #
def translate_to_english(text):
    """把中文文本翻译成英文。需要环境变量 DEEPSEEK_API_KEY；失败时抛出 RuntimeError。"""
    text = (text or "").strip()
    if not text:
        return ""
    import requests  # 惰性导入，仅 --prompt_with_text 时才需要

    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("使用 --prompt_with_text 需要设置环境变量 DEEPSEEK_API_KEY")

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            json={
                "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
                "messages": [
                    {"role": "system",
                     "content": "你是专业翻译。把下面的中文翻译成英文，只输出英文译文，不要任何解释、前缀或引号。"},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.3,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Connection": "close",
            },
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        raise RuntimeError(f"翻译失败: {e}") from e

    if not out:
        raise RuntimeError("翻译结果为空")
    return out


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _exists_nonempty(path):
    """文件存在且非空。"""
    return Path(path).is_file() and Path(path).stat().st_size > 0


def _parse_num_arg(value):
    """argparse 类型：把 '3' 或 'scene3' 解析成 int。"""
    m = re.search(r"\d+", str(value))
    if not m:
        raise argparse.ArgumentTypeError(f"无法解析数字: {value!r}")
    return int(m.group())


def add_filter_args(parser):
    """给解析器添加 --scene / --shot 两个通用筛选参数。"""
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只处理指定场景（数字，如 --scene 3）")
    parser.add_argument("--shot", type=_parse_num_arg, default=None,
                        help="只处理指定场景内的指定分镜（需与 --scene 配合，如 --scene 3 --shot 2）")
