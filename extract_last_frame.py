#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抽取视频的最后一帧图像。

用法:
    python extract_last_frame.py <视频路径> <图像路径>

依赖:
    ffmpeg / ffprobe（需在 PATH 中，或通过环境变量 FFMPEG / FFPROBE 指定）

示例:
    python extract_last_frame.py input.mp4 output.jpg
"""

import argparse
import os
import shutil
import subprocess
import sys

# Windows 控制台默认编码可能是 cp1252/gbk，统一用 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def which_or_env(name, env_var):
    """优先使用环境变量指定的可执行文件，否则回退到 PATH 查找。"""
    exe = os.environ.get(env_var)
    if exe:
        return exe
    found = shutil.which(name)
    if found:
        return found
    return name  # 交给 subprocess 报错


FFMPEG = which_or_env("ffmpeg", "FFMPEG")
FFPROBE = which_or_env("ffprobe", "FFPROBE")


def get_total_frames(video_path):
    """用 ffprobe 精确统计视频总帧数。"""
    cmd = [
        FFPROBE, "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    value = result.stdout.strip()
    if value.isdigit():
        return int(value)
    raise RuntimeError(
        f"无法读取视频帧数: {result.stderr.strip() or result.stdout.strip()}"
    )


def extract_frame(video_path, image_path, frame_index):
    """抽取指定索引（从 0 开始）的一帧。"""
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"select=eq(n\\,{frame_index})",
        "-frames:v", "1",
        "-y", image_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())


def main():
    parser = argparse.ArgumentParser(description="抽取视频的最后一帧图像")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("image", help="输出图像路径")
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    image_path = os.path.abspath(args.image)

    if not os.path.isfile(video_path):
        print(f"错误: 输入视频不存在: {video_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(image_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    try:
        total = get_total_frames(video_path)
        if total <= 0:
            raise RuntimeError("视频帧数为 0，可能不是有效的视频文件")
        last_index = total - 1
        print(f"视频共 {total} 帧，正在抽取第 {last_index} 帧（最后一帧）...")
        extract_frame(video_path, image_path, last_index)
    except RuntimeError as e:
        print(f"抽取失败: {e}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(image_path):
        print("抽取失败: 未生成图像文件", file=sys.stderr)
        sys.exit(1)

    print(f"已抽取最后一帧: {image_path}")


if __name__ == "__main__":
    main()
