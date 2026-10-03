#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把一段配音 mux 进视频（视频流无损拷贝，音频流 AAC）。

常与 concat_clips.py（拼出 result.mp4）和 concat_audio.py（拼出 work/result_audio.m4a）
配合使用。

用法：
  python add_audio.py <视频路径> <音频路径> [--output 输出路径]

  --output  输出路径（默认 <视频名>_with_audio.mp4）。

示例：
  python add_audio.py result.mp4 work/result_audio.m4a
"""

import argparse
from pathlib import Path

from ltx_common import mux_audio


def main():
    parser = argparse.ArgumentParser(description="把配音合成进视频")
    parser.add_argument("video", help="输入视频路径")
    parser.add_argument("audio", help="输入音频路径")
    parser.add_argument("--output", default=None,
                        help="输出路径（默认 <视频名>_with_audio.mp4）")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    audio = Path(args.audio).resolve()
    if not video.is_file():
        raise RuntimeError(f"输入视频不存在：{video}")
    if not audio.is_file():
        raise RuntimeError(f"输入音频不存在：{audio}")

    if args.output:
        output = Path(args.output)
    else:
        output = video.with_name(video.stem + "_with_audio.mp4")

    print(f"输入视频 : {video}")
    print(f"输入音频 : {audio}")
    print(f"输出路径 : {output}")
    print("=" * 70)
    print(f"正在合成配音 -> {output} ...", flush=True)
    mux_audio(video, audio, output)
    print(f"已生成：{output}")


if __name__ == "__main__":
    main()
