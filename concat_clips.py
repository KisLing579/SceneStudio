#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 clips/ 下的分镜视频按 meta.json 顺序拼接为结果视频（纯视频，不含配音）。

按 meta.json 的 prompts 顺序读取 output_{shot_idx}.mp4，用 ffmpeg concat 无损拼接。
默认输出 result.mp4；可用 --output 覆盖，用 --scene / --shot 只拼接指定范围内的分镜。

用法：
  python concat_clips.py [meta.json 路径] [--scene N] [--shot M] [--output 输出路径]

  --scene N    只拼接第 N 个场景的分镜（如 --scene 3）。
  --shot M     只拼接指定场景内的第 M 个分镜（需与 --scene 配合）。
  --output     输出路径（默认 result.mp4）。
"""

import argparse
from pathlib import Path

from ltx_common import (
    CLIPS_DIR,
    RESULT_PATH,
    load_meta,
    concat_videos,
    _exists_nonempty,
    _parse_num_arg,
    select_indices,
)


def main():
    parser = argparse.ArgumentParser(description="把 clips/ 分镜视频拼接为结果视频")
    parser.add_argument("meta", nargs="?", default="meta.json",
                        help="meta.json 路径（用于确定顺序与筛选）")
    parser.add_argument("--scene", type=_parse_num_arg, default=None,
                        help="只拼接指定场景（数字，如 --scene 3）")
    parser.add_argument("--shot", type=_parse_num_arg, default=None,
                        help="只拼接指定场景内的指定分镜（数字，需与 --scene 配合）")
    parser.add_argument("--output", default=str(RESULT_PATH),
                        help=f"输出路径（默认 {RESULT_PATH}）")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    prompts, *_ = load_meta(meta_path)
    indices = select_indices(prompts, args.scene, args.shot)

    clips = []
    for idx in indices:
        shot_idx = prompts[idx].get("shot_idx", idx)
        p = CLIPS_DIR / f"output_{shot_idx}.mp4"
        if not _exists_nonempty(p):
            raise RuntimeError(f"缺少分镜视频（请先运行 generate_clips.py）：{p}")
        clips.append(p)

    output = Path(args.output)
    if args.scene is not None or args.shot is not None:
        cond = []
        if args.scene is not None:
            cond.append(f"场景 {args.scene}")
        if args.shot is not None:
            cond.append(f"分镜 {args.shot}")
        print(f"筛选条件 : {'，'.join(cond)}")

    print(f"分镜目录 : {CLIPS_DIR}")
    print(f"待拼接数 : {len(clips)}")
    print(f"输出路径 : {output}")
    print("=" * 70)
    print(f"正在拼接 {len(clips)} 个分镜视频 -> {output} ...", flush=True)
    concat_videos(clips, output)
    print(f"结果视频已生成：{output}")


if __name__ == "__main__":
    main()
