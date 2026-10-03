#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 meta.json 生成 key_frame_flags（关键帧标记）布尔数组。

规则（同一场景内，按 prompts 顺序遍历）：
  - 每个场景的第一个分镜 -> True；
  - 其余分镜：若当前分镜的 focus_characters 里出现了该场景上一镜没有的新元素
    （即有新角色进场），则 True，否则 False。

shot_id 由 scene{场景编号}-shot{分镜编号} 组成；场景分组优先取 scene_idx，
缺省时从 shot_id 解析 scene 编号。

结果写入 meta.json 的 task_data.key_frame_flags（与 prompts/images 并列）。

用法：
  python generate_key_frame_flags.py [meta.json 路径]
默认读取 ./work/meta.json。
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Windows 控制台统一 UTF-8 输出，避免中文报错
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def scene_key(p):
    """返回场景分组键：优先 scene_idx，其次从 shot_id 解析 scene 编号。"""
    si = p.get("scene_idx")
    if si is not None:
        return si
    m = re.search(r"scene(\d+)", str(p.get("shot_id") or ""), re.IGNORECASE)
    return int(m.group(1)) if m else None


def build_key_frame_flags(prompts):
    """按规则生成与 prompts 等长的布尔数组。"""
    flags = []
    seen_scenes = set()
    prev_fc = {}  # scene_key -> 该场景上一个分镜的 focus_characters 元素集合
    for p in prompts:
        sc = scene_key(p)
        fc = p.get("focus_characters") or []
        if sc not in seen_scenes:
            # 场景的第一个分镜
            flags.append(True)
            seen_scenes.add(sc)
        else:
            prev = prev_fc.get(sc, set())
            # 出现上一镜没有的新元素（新角色进场）则 True
            flags.append(any(x not in prev for x in fc))
        prev_fc[sc] = set(fc)
    return flags


def main():
    parser = argparse.ArgumentParser(description="为 meta.json 生成 key_frame_flags（关键帧标记）")
    parser.add_argument("meta", nargs="?", default="./work/meta.json",
                        help="meta.json 路径（默认 ./work/meta.json）")
    args = parser.parse_args()

    meta_path = Path(args.meta).resolve()
    if not meta_path.is_file():
        raise SystemExit(f"找不到 meta.json：{meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task = data.get("task_data") or data
    prompts = task.get("prompts") or []
    if not prompts:
        raise SystemExit("meta.json 中未找到 prompts 列表")

    flags = build_key_frame_flags(prompts)
    if len(flags) != len(prompts):
        raise SystemExit(f"内部错误：flags 长度({len(flags)}) 与 prompts 长度({len(prompts)}) 不一致")

    task["key_frame_flags"] = flags

    # 与现有 meta.json 保持一致：indent=2、ensure_ascii=False、CRLF、无末尾换行
    text = json.dumps(data, ensure_ascii=False, indent=2)
    text = text.replace("\n", "\r\n")
    with open(meta_path, "wb") as f:
        f.write(text.encode("utf-8"))

    true_count = sum(1 for f in flags if f)
    print(f"已写入 {len(flags)} 个 key_frame_flags 到 {meta_path}")
    print(f"关键帧数：{true_count} / {len(flags)}\n")

    # 按场景打印明细，便于核对
    prev_scene = None
    for i, (p, f) in enumerate(zip(prompts, flags)):
        sc = scene_key(p)
        if sc != prev_scene:
            print(f"场景 {sc}:")
            prev_scene = sc
        fc = p.get("focus_characters") or []
        print(f"  [{i:>2}] {p.get('shot_id')}  {fc}  ->  {f}")


if __name__ == "__main__":
    main()
