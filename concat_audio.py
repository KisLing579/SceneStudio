#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 meta.json 的 audios 按顺序拼接为一段 AAC 配音。

按 prompts 顺序读取 audios（相对 meta.json 所在目录），用 ffmpeg concat 拼接。
默认输出 work/result_audio.m4a；可用 --output 覆盖，用 --scene / --shot 只拼接指定范围。

用法：
  python concat_audio.py [meta.json 路径] [--scene N] [--shot M] [--output 输出路径]

  --scene N    只拼接第 N 个场景的配音（如 --scene 3）。
  --shot M     只拼接指定场景内的第 M 个分镜配音（需与 --scene 配合）。
  --output     输出路径（默认 work/result_audio.m4a）。
"""

import argparse
from pathlib import Path

from ltx_common import (
    AUDIO_CONCAT_PATH,
    load_meta,
    concat_audios,
    _parse_num_arg,
    select_indices,
)


def main():
    parser = argparse.ArgumentParser(description="把 meta.json 的 audios 拼接为一段配音")
    parser.add_argument("meta", nargs="?", default="meta.json",
                        help="meta.json 路径（用于确定顺序与筛选）")
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只拼接指定场景（数字，如 --scene 3）")
    parser.add_argument("--shot", type=_parse_num_arg, default=None,
                        help="只拼接指定场景内的指定分镜（数字，需与 --scene 配合）")
    parser.add_argument("--output", default=str(AUDIO_CONCAT_PATH),
                        help=f"输出路径（默认 {AUDIO_CONCAT_PATH}）")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    prompts, _, _, audios, _ = load_meta(meta_path)
    base_dir = meta_path.parent
    indices = select_indices(prompts, args.scene, args.shot)

    if not audios:
        raise RuntimeError("meta.json 中未找到 audios 列表")

    audio_paths = []
    for idx in indices:
        a = (base_dir / audios[idx]).resolve()
        if not a.is_file():
            raise RuntimeError(f"缺少配音文件（相对 meta.json 目录）：{audios[idx]}")
        audio_paths.append(a)

    output = Path(args.output)
    if args.scene is not None or args.shot is not None:
        cond = []
        if args.scene is not None:
            cond.append(f"场景 {args.scene}")
        if args.shot is not None:
            cond.append(f"分镜 {args.shot}")
        print(f"筛选条件 : {'，'.join(cond)}")

    print(f"待拼接数 : {len(audio_paths)}")
    print(f"输出路径 : {output}")
    print("=" * 70)
    print(f"正在拼接 {len(audio_paths)} 段配音 -> {output} ...", flush=True)
    concat_audios(audio_paths, output)
    print(f"配音已生成：{output}")


if __name__ == "__main__":
    main()
